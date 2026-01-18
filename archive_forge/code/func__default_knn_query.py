from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def _default_knn_query(self, query_vector: Optional[List[float]]=None, query: Optional[str]=None, model_id: Optional[str]=None, k: Optional[int]=10, num_candidates: Optional[int]=10) -> Dict:
    knn: Dict = {'field': self.vector_query_field, 'k': k, 'num_candidates': num_candidates}
    if query_vector and (not model_id):
        knn['query_vector'] = query_vector
    elif query and model_id:
        knn['query_vector_builder'] = {'text_embedding': {'model_id': model_id, 'model_text': query}}
    else:
        raise ValueError('Either `query_vector` or `model_id` must be provided, but not both.')
    return knn