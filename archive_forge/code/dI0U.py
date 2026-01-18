import duckdb
import pandas as pd
import logging
from typing import Dict, Tuple

import torch
from transformers import BertModel, BertTokenizer
from typing import Dict
import logging

import streamlit as st


class DatabaseManager:
    """
    Manages database operations for document storage and retrieval.

    Attributes:
        con (duckdb.DuckDBPyConnection): The connection to the database.
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize the database connection.

        Args:
            db_path (str): Path to the database file, defaults to in-memory database.
        """
        try:
            self.con = duckdb.connect(database=db_path, read_only=False)
            self.setup_database()
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            raise

    def setup_database(self) -> None:
        """
        Sets up the database schema for document storage.
        """
        try:
            self.con.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER,
                    text VARCHAR,
                    embedding BLOB,
                    metadata JSON,
                    creation_date DATE DEFAULT CURRENT_DATE,
                    tags JSON,
                    PRIMARY KEY (id)
                );
            """
            )
            logging.info("Database schema setup successfully.")
        except Exception as e:
            logging.error(f"Failed to setup database schema: {e}")
            raise

    def insert_document(
        self, doc_id: int, text: str, embedding: bytes, metadata: Dict
    ) -> None:
        """
        Inserts a document along with its metadata into the database.

        Args:
            doc_id (int): Document identifier.
            text (str): Document text.
            embedding (bytes): BERT model generated embedding of the document.
            metadata (Dict): Metadata associated with the document.
        """
        try:
            self.con.execute(
                "INSERT INTO documents (id, text, embedding, metadata) VALUES (?, ?, ?, ?)",
                (doc_id, text, embedding, metadata),
            )
            logging.info(f"Document {doc_id} inserted successfully.")
        except Exception as e:
            logging.error(f"Failed to insert document {doc_id}: {e}")
            raise

    def search_documents(self, query_embedding: bytes) -> pd.DataFrame:
        """
        Searches for documents similar to the query embedding.

        Args:
            query_embedding (bytes): BERT model generated embedding of the query text.

        Returns:
            pd.DataFrame: A dataframe containing the top 10 similar documents.
        """
        try:
            result = self.con.execute(
                """
                SELECT id, text, cosine_similarity(embedding, ?) AS similarity
                FROM documents
                ORDER BY similarity DESC
                LIMIT 10
                """,
                (query_embedding,),
            ).fetchdf()
            return result
        except Exception as e:
            logging.error("Failed to search documents: {e}")
            raise


class EmbeddingManager:
    """
    Handles embedding generation and text analysis using BERT model.

    Attributes:
        tokenizer (BertTokenizer): Tokenizer for BERT model.
        model (BertModel): Pre-trained BERT model.
    """

    def __init__(self):
        """
        Initialize the BERT tokenizer and model.
        """
        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            logging.error(f"Failed to load BERT model: {e}")
            raise

    def generate_embeddings(self, text: str) -> torch.Tensor:
        """
        Generates embeddings for the input text using BERT.

        Args:
            text (str): Text to generate embeddings for.

        Returns:
            torch.Tensor: The generated embeddings as a PyTorch tensor.
        """
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]
        except Exception as e:
            logging.error(f"Failed to generate embeddings for text: {e}")
            raise

    def text_analysis(self, text: str) -> Dict:
        """
        Performs text analysis to extract tokens, remove stopwords, and tag parts of speech.

        Args:
            text (str): Text to analyze.

        Returns:
            Dict: A dictionary containing tokens, stopwords removed, and parts of speech tags.
        """
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag

            tokens = word_tokenize(text)
            stopwords_removed = [
                word
                for word in tokens
                if word.lower() not in stopwords.words("english")
            ]
            pos_tags = pos_tag(stopwords_removed)
            return {
                "tokens": tokens,
                "stopwords_removed": stopwords_removed,
                "pos_tags": pos_tags,
            }
        except Exception as e:
            logging.error(f"Failed to perform text analysis on text: {e}")
            raise


def initialize_streamlit_interface() -> None:
    """
    Initializes and configures the Streamlit web interface.
    """
    st.title("Semantic Search System")

    # Database and embedding managers
    db_manager = DatabaseManager()
    embedding_manager = EmbeddingManager()

    # Document insertion interface
    with st.form("add_document"):
        doc_id = st.number_input("Document ID", min_value=1, value=1, step=1)
        text = st.text_area("Document Text")
        tags = st.text_input("Document Tags (comma-separated)")
        submitted = st.form_submit_button("Insert Document")
        if submitted and text:
            try:
                embedding = embedding_manager.generate_embeddings(text)
                metadata = embedding_manager.text_analysis(text)
                db_manager.insert_document(
                    doc_id,
                    text,
                    embedding.numpy().tobytes(),
                    {"analysis": metadata, "tags": tags.split(",")},
                )
                st.success("Document inserted successfully!")
            except Exception as e:
                st.error(f"Error inserting document: {e}")

    # Document search interface
    with st.form("search"):
        query = st.text_area("Search Query")
        search_submitted = st.form_submit_button("Search")
        if search_submitted and query:
            try:
                query_embedding = embedding_manager.generate_embeddings(query)
                results = db_manager.search_documents(query_embedding.numpy().tobytes())
                st.write(results)
            except Exception as e:
                st.error(f"Error searching documents: {e}")

    st.button("Re-run")


if __name__ == "__main__":
    initialize_streamlit_interface()
