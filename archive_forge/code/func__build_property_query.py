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
def _build_property_query(collection_name: str, command_type: str='find', all_properties: List=[], ref: Optional[int]=None) -> Tuple[Any, Any]:
    all_queries: List[Any] = []
    blob_arr: List[Any] = []
    choices = ['find', 'add', 'update']
    if command_type.lower() not in choices:
        raise ValueError('[!] Invalid type. Choices are : {}'.format(','.join(choices)))
    if command_type.lower() == 'find':
        query = _find_property_entity(collection_name, unique_entity=True)
        all_queries.append(query)
    elif command_type.lower() == 'add':
        query, byte_data = _add_entity_with_blob(collection_name, all_properties)
        all_queries.append(query)
        blob_arr.append(byte_data)
    elif command_type.lower() == 'update':
        query = _find_property_entity(collection_name, deletion=True)
        all_queries.append(query)
        query, byte_data = _add_entity_with_blob(collection_name, all_properties)
        all_queries.append(query)
        blob_arr.append(byte_data)
    return (all_queries, blob_arr)