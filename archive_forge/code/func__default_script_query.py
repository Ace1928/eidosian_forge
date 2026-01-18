from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def _default_script_query(query_vector: List[float], filter: Optional[dict]) -> Dict:
    if filter:
        (key, value), = filter.items()
        filter = {'match': {f'metadata.{key}.keyword': f'{value}'}}
    else:
        filter = {'match_all': {}}
    return {'script_score': {'query': filter, 'script': {'source': "cosineSimilarity(params.query_vector, 'vector') + 1.0", 'params': {'query_vector': query_vector}}}}