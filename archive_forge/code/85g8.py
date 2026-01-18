import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from transformers import BertModel, BertTokenizer
import torch
import duckdb
import pandas as pd
import streamlit as st

# Optimize imports and ensure clear module usage
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Preload resources to improve performance
stop_words = set(stopwords.words("english"))


# Singleton class to manage BERT resources efficiently
class BERTResourceManager:
    """Manages loading and usage of BERT model and tokenizer to optimize resources."""

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    @staticmethod
    def generate_embeddings(text: str) -> torch.Tensor:
        """Generates BERT embeddings for the given text."""
        inputs = BERTResourceManager.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = BERTResourceManager.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Use the CLS token's embedding


# Database initialization and management
def initialize_database() -> duckdb.DuckDBPyConnection:
    """Initializes and returns a connection to an in-memory DuckDB database with predefined schema."""
    connection = duckdb.connect(database=":memory:", read_only=False)
    connection.execute(
        """
        CREATE TABLE documents (
            id INTEGER,
            text VARCHAR,
            embedding BLOB,
            metadata JSON
        )
    """
    )
    return connection


class DocumentManager:
    """Encapsulates database operations for document management including insertion and search."""

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.con = connection

    def insert_document(self, doc_id: int, text: str):
        """Inserts a document into the database with its embedding and metadata."""
        embedding = BERTResourceManager.generate_embeddings(text)
        analysis = self.perform_text_analysis(text)
        self.con.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?)",
            (doc_id, text, embedding.tobytes(), {"analysis": analysis}),
        )

    def search_documents(self, query: str) -> pd.DataFrame:
        """Searches documents by similarity to the query and returns the top matches."""
        query_embedding = BERTResourceManager.generate_embeddings(query)
        result = self.con.execute(
            """
            SELECT id, text, cosine_similarity(embedding, ?) AS similarity
            FROM documents
            ORDER BY similarity DESC
            LIMIT 10
            """,
            (query_embedding.tobytes(),),
        ).fetchdf()
        return result

    @staticmethod
    def perform_text_analysis(text: str) -> dict:
        """Analyzes text to extract tokens, remove stopwords, and tag parts of speech."""
        tokens = word_tokenize(text)
        stopwords_removed = [word for word in tokens if word.lower() not in stop_words]
        pos_tags = pos_tag(stopwords_removed)
        return {
            "tokens": tokens,
            "stopwords_removed": stopwords_removed,
            "pos_tags": pos_tags,
        }


# Streamlit application setup and UI components
def main():
    st.title("Semantic Search with DuckDB")

    # Initialize the database and manager
    con = initialize_database()
    manager = DocumentManager(con)

    # Interface to add documents
    with st.form("add_document"):
        doc_id = st.number_input("Document ID", min_value=1, value=1, step=1)
        text = st.text_area("Document Text")
        submitted = st.form_submit_button("Insert Document")
        if submitted:
            manager.insert_document(doc_id, text)
            st.success("Document inserted successfully!")

    # Interface to perform search
    with st.form("search"):
        query = st.text_input("Search Query")
        search_submitted = st.form_submit_button("Search")
        if search_submitted:
            results = manager.search_documents(query)
            st.write(results)

    st.button("Re-run")


if __name__ == "__main__":
    main()
