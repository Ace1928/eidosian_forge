from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _prepare_vector_query(self, k: int, filter: Optional[RedisFilterExpression]=None, return_fields: Optional[List[str]]=None) -> 'Query':
    """Prepare query for vector search.

        Args:
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            query: Query object.
        """
    try:
        from redis.commands.search.query import Query
    except ImportError as e:
        raise ImportError('Could not import redis python package. Please install it with `pip install redis`.') from e
    return_fields = return_fields or []
    query_prefix = '*'
    if filter:
        query_prefix = f'{str(filter)}'
    vector_key = self._schema.content_vector_key
    base_query = f'({query_prefix})=>[KNN {k} @{vector_key} $vector AS distance]'
    query = Query(base_query).return_fields(*return_fields).sort_by('distance').paging(0, k).dialect(2)
    return query