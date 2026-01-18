import sqlite3
from typing import List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
from IntelligentVectorDB.file_processor import (
    FileProcessor,
)  # Importing the FileProcessor class


class EmbeddingStorage:
    """
    Manages storage and retrieval of text embeddings, using the mlpnet_base_v2 model for embedding generation.
    This class uses an SQLite database for local storage of embeddings to ensure that operations can be performed offline.
    It integrates with the FileProcessor to process and embed text from various document types.
    """

    def __init__(self, db_path: str, folder_path: str):
        """
        Initializes the database connection and sets up the file processing.
        :param db_path: str - Path to the SQLite database file.
        :param folder_path: str - Path to the folder containing documents to be processed and embedded.
        """
        self.db_path = db_path
        self.folder_path = folder_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.file_processor = FileProcessor(
            self.folder_path
        )  # Initialize the FileProcessor
        self._create_tables()

    def _create_tables(self) -> None:
        """
        Creates the necessary tables in the database if they do not already exist.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                embedding BLOB
            )
        """
        )
        self.connection.commit()

    def _generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generates embeddings for the given text using the mlpnet_base_v2 model.
        :param text: str - Text to be embedded.
        :return: np.ndarray - The generated embeddings.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/mlpnet-base-v2"
        )
        model = AutoModel.from_pretrained("sentence-transformers/mlpnet-base-v2")

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def store_embeddings(self) -> None:
        """
        Processes documents from the specified folder, generates embeddings for the extracted text,
        and stores them in the database.
        """
        text_data = (
            self.file_processor.process_files()
        )  # Extract text data using FileProcessor
        for file_path, text in text_data:
            embedding = self._generate_embeddings(text)
            self.cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings (file_path, embedding) VALUES (?, ?)
            """,
                (file_path, embedding.tobytes()),
            )
        self.connection.commit()

    def load_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieves all embeddings from the database, converting them back into numpy arrays.
        :return: List[Tuple[str, np.ndarray]] - A list of tuples containing file paths and their numpy array embeddings.
        """
        self.cursor.execute("SELECT file_path, embedding FROM embeddings")
        data = self.cursor.fetchall()
        return [
            (file_path, np.frombuffer(embedding, dtype=np.float32))
            for file_path, embedding in data
        ]

    def __del__(self):
        """
        Ensures the database connection is closed when the object is deleted.
        """
        self.connection.close()
