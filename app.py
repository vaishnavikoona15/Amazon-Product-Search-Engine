import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image


# Load the dataset
data = pd.read_csv("amazon_product.csv")
data.head()

data.drop("id", axis=1, inplace=True)
data.head()

# Handling null values
data.isnull().sum()


# NLP
stemmer = SnowballStemmer('english')


def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)


data["stemmed_tokens"] = data.apply(lambda row: tokenize_stem(row["Title"] + " " + row["Description"]), axis=1)
data.head()


tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_stem)


def cosine_sim(txt1, txt2):
    txt1_concatenated = " ".join(txt1)
    txt2_concatenated = " ".join(txt2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([txt1_concatenated, txt2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]


def search_product(query):
    stemmed_query = tokenize_stem(query)

    # Calculating cosine similarity between query and stemmed tokens columns
    data["similarity"] = data["stemmed_tokens"].apply(lambda x: cosine_sim(stemmed_query, x))
    results = data.sort_values(by=["similarity"], ascending=False).head(10)[["Title", "Description", "Category"]]
    return results


# WEB PAGE

img = Image.open("img.png")
st.image(img, width=600)
st.title("Search Engine and Product Recommendation System on Amazon Data")
query = st.text_input("Enter Product Name")
submit = st.button("Search")
if submit:
    res = search_product(query)
    st.write(res)
