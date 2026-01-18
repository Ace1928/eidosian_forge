import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
import numpy as np
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
class BaseRetrievalStrategy(ABC):
    """Base class for `Elasticsearch` retrieval strategies."""

    @abstractmethod
    def query(self, query_vector: Union[List[float], None], query: Union[str, None], *, k: int, fetch_k: int, vector_query_field: str, text_field: str, filter: List[dict], similarity: Union[DistanceStrategy, None]) -> Dict:
        """
        Executes when a search is performed on the store.

        Args:
            query_vector: The query vector,
                          or None if not using vector-based query.
            query: The text query, or None if not using text-based query.
            k: The total number of results to retrieve.
            fetch_k: The number of results to fetch initially.
            vector_query_field: The field containing the vector
                                representations in the index.
            text_field: The field containing the text data in the index.
            filter: List of filter clauses to apply to the query.
            similarity: The similarity strategy to use, or None if not using one.

        Returns:
            Dict: The Elasticsearch query body.
        """

    @abstractmethod
    def index(self, dims_length: Union[int, None], vector_query_field: str, similarity: Union[DistanceStrategy, None]) -> Dict:
        """
        Executes when the index is created.

        Args:
            dims_length: Numeric length of the embedding vectors,
                        or None if not using vector-based query.
            vector_query_field: The field containing the vector
                                representations in the index.
            similarity: The similarity strategy to use,
                        or None if not using one.

        Returns:
            Dict: The Elasticsearch settings and mappings for the strategy.
        """

    def before_index_setup(self, client: 'Elasticsearch', text_field: str, vector_query_field: str) -> None:
        """
        Executes before the index is created. Used for setting up
        any required Elasticsearch resources like a pipeline.

        Args:
            client: The Elasticsearch client.
            text_field: The field containing the text data in the index.
            vector_query_field: The field containing the vector
                                representations in the index.
        """

    def require_inference(self) -> bool:
        """
        Returns whether or not the strategy requires inference
        to be performed on the text before it is added to the index.

        Returns:
            bool: Whether or not the strategy requires inference
            to be performed on the text before it is added to the index.
        """
        return True