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
class ApproxRetrievalStrategy(BaseRetrievalStrategy):
    """Approximate retrieval strategy using the `HNSW` algorithm."""

    def __init__(self, query_model_id: Optional[str]=None, hybrid: Optional[bool]=False, rrf: Optional[Union[dict, bool]]=True):
        self.query_model_id = query_model_id
        self.hybrid = hybrid
        self.rrf = rrf

    def query(self, query_vector: Union[List[float], None], query: Union[str, None], k: int, fetch_k: int, vector_query_field: str, text_field: str, filter: List[dict], similarity: Union[DistanceStrategy, None]) -> Dict:
        knn = {'filter': filter, 'field': vector_query_field, 'k': k, 'num_candidates': fetch_k}
        if query_vector and (not self.query_model_id):
            knn['query_vector'] = query_vector
        elif query and self.query_model_id:
            knn['query_vector_builder'] = {'text_embedding': {'model_id': self.query_model_id, 'model_text': query}}
        else:
            raise ValueError('You must provide an embedding function or a query_model_id to perform a similarity search.')
        if self.hybrid:
            query_body = {'knn': knn, 'query': {'bool': {'must': [{'match': {text_field: {'query': query}}}], 'filter': filter}}}
            if isinstance(self.rrf, dict):
                query_body['rank'] = {'rrf': self.rrf}
            elif isinstance(self.rrf, bool) and self.rrf is True:
                query_body['rank'] = {'rrf': {}}
            return query_body
        else:
            return {'knn': knn}

    def index(self, dims_length: Union[int, None], vector_query_field: str, similarity: Union[DistanceStrategy, None]) -> Dict:
        """Create the mapping for the Elasticsearch index."""
        if similarity is DistanceStrategy.COSINE:
            similarityAlgo = 'cosine'
        elif similarity is DistanceStrategy.EUCLIDEAN_DISTANCE:
            similarityAlgo = 'l2_norm'
        elif similarity is DistanceStrategy.DOT_PRODUCT:
            similarityAlgo = 'dot_product'
        elif similarity is DistanceStrategy.MAX_INNER_PRODUCT:
            similarityAlgo = 'max_inner_product'
        else:
            raise ValueError(f'Similarity {similarity} not supported.')
        return {'mappings': {'properties': {vector_query_field: {'type': 'dense_vector', 'dims': dims_length, 'index': True, 'similarity': similarityAlgo}}}}