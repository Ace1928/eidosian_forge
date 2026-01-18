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
@deprecated('0.0.27', alternative='Use class in langchain-elasticsearch package', pending=True)
class ExactRetrievalStrategy(BaseRetrievalStrategy):
    """Exact retrieval strategy using the `script_score` query."""

    def query(self, query_vector: Union[List[float], None], query: Union[str, None], k: int, fetch_k: int, vector_query_field: str, text_field: str, filter: Union[List[dict], None], similarity: Union[DistanceStrategy, None]) -> Dict:
        if similarity is DistanceStrategy.COSINE:
            similarityAlgo = f"cosineSimilarity(params.query_vector, '{vector_query_field}') + 1.0"
        elif similarity is DistanceStrategy.EUCLIDEAN_DISTANCE:
            similarityAlgo = f"1 / (1 + l2norm(params.query_vector, '{vector_query_field}'))"
        elif similarity is DistanceStrategy.DOT_PRODUCT:
            similarityAlgo = f"\n            double value = dotProduct(params.query_vector, '{vector_query_field}');\n            return sigmoid(1, Math.E, -value);\n            "
        else:
            raise ValueError(f'Similarity {similarity} not supported.')
        queryBool: Dict = {'match_all': {}}
        if filter:
            queryBool = {'bool': {'filter': filter}}
        return {'query': {'script_score': {'query': queryBool, 'script': {'source': similarityAlgo, 'params': {'query_vector': query_vector}}}}}

    def index(self, dims_length: Union[int, None], vector_query_field: str, similarity: Union[DistanceStrategy, None]) -> Dict:
        """Create the mapping for the Elasticsearch index."""
        return {'mappings': {'properties': {vector_query_field: {'type': 'dense_vector', 'dims': dims_length, 'index': False}}}}