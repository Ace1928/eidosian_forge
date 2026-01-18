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
def _default_painless_scripting_query(query_vector: List[float], k: int=4, space_type: str='l2Squared', pre_filter: Optional[Dict]=None, vector_field: str='vector_field') -> Dict:
    """For Painless Scripting Search, this is the default query."""
    if not pre_filter:
        pre_filter = MATCH_ALL_QUERY
    source = __get_painless_scripting_source(space_type, vector_field=vector_field)
    return {'size': k, 'query': {'script_score': {'query': pre_filter, 'script': {'source': source, 'params': {'field': vector_field, 'query_value': query_vector}}}}}