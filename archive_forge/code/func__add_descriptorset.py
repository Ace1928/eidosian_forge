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
def _add_descriptorset(command_str: str, name: str, num_dims: Optional[int]=None, engine: Optional[str]=None, metric: Optional[str]=None, ref: Optional[int]=None, props: Optional[Dict]=None, link: Optional[Dict]=None, storeIndex: bool=False, constraints: Optional[Dict]=None, results: Optional[Dict]=None) -> Dict[str, Any]:
    if command_str == 'AddDescriptorSet' and all((var is not None for var in [name, num_dims])):
        entity: Dict[str, Any] = {'name': name, 'dimensions': num_dims}
        if engine is not None:
            entity['engine'] = engine
        if metric is not None:
            entity['metric'] = metric
        if ref is not None:
            entity['_ref'] = ref
        if props not in [None, {}]:
            entity['properties'] = props
        if link is not None:
            entity['link'] = link
    elif command_str == 'FindDescriptorSet':
        entity = {'set': name}
        if storeIndex:
            entity['storeIndex'] = storeIndex
        if constraints not in [None, {}]:
            entity['constraints'] = constraints
        if results is not None:
            entity['results'] = results
    else:
        raise ValueError(f'Unknown command: {command_str}')
    query = {command_str: entity}
    return query