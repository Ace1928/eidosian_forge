from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def __delete(self, collection_name: str, ids: Union[None, List[str]]=None, constraints: Union[None, Dict[str, Any]]=None) -> bool:
    """
        Deletes entire collection if id is not provided
        """
    all_queries: List[Any] = []
    all_blobs: List[Any] = []
    collection_properties = self.__get_properties(collection_name)
    results = {'list': collection_properties}
    if constraints is None:
        constraints = {'_deletion': ['==', 1]}
    else:
        constraints['_deletion'] = ['==', 1]
    if ids is not None:
        constraints['id'] = ['==', ids[0]]
    query = _add_descriptor('FindDescriptor', collection_name, label=None, ref=None, props=None, link=None, k_neighbors=None, constraints=constraints, results=results)
    all_queries.append(query)
    response, response_array = self.__run_vdms_query(all_queries, all_blobs)
    return 'FindDescriptor' in response[0]