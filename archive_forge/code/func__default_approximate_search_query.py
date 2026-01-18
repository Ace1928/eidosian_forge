from __future__ import annotations
import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _default_approximate_search_query(query_vector: List[float], k: int=4, vector_field: str='vector_field') -> Dict:
    """For Approximate k-NN Search, this is the default query."""
    return {'size': k, 'query': {'knn': {vector_field: {'vector': query_vector, 'k': k}}}}