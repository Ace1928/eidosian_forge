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
def __get_add_query(self, collection_name: str, metadata: Optional[Any]=None, embedding: Union[List[float], None]=None, document: Optional[Any]=None, id: Optional[str]=None) -> Tuple[Dict[str, Dict[str, Any]], Union[bytes, None]]:
    if id is None:
        props: Dict[str, Any] = {}
    else:
        props = {'id': id}
        id_exists, query = _check_descriptor_exists_by_id(self._client, collection_name, id)
        if id_exists:
            skipped_value = {prop_key: prop_val[-1] for prop_key, prop_val in query['FindDescriptor']['constraints'].items()}
            pstr = f'[!] Embedding with id ({id}) exists in DB;'
            pstr += 'Therefore, skipped and not inserted'
            print(pstr)
            print(f'\tSkipped values are: {skipped_value}')
            return (query, None)
    if metadata:
        props.update(metadata)
    if document:
        props['content'] = document
    for k in props.keys():
        if k not in self.collection_properties:
            self.collection_properties.append(k)
    query = _add_descriptor('AddDescriptor', collection_name, label=None, ref=None, props=props, link=None, k_neighbors=None, constraints=None, results=None)
    blob = embedding2bytes(embedding)
    return (query, blob)