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
def _bulk_ingest_embeddings(client: Any, index_name: str, embeddings: List[List[float]], texts: Iterable[str], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, vector_field: str='vector_field', text_field: str='text', mapping: Optional[Dict]=None, max_chunk_bytes: Optional[int]=1 * 1024 * 1024, is_aoss: bool=False) -> List[str]:
    """Bulk Ingest Embeddings into given index."""
    if not mapping:
        mapping = dict()
    bulk = _import_bulk()
    not_found_error = _import_not_found_error()
    requests = []
    return_ids = []
    mapping = mapping
    try:
        client.indices.get(index=index_name)
    except not_found_error:
        client.indices.create(index=index_name, body=mapping)
    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas else {}
        _id = ids[i] if ids else str(uuid.uuid4())
        request = {'_op_type': 'index', '_index': index_name, vector_field: embeddings[i], text_field: text, 'metadata': metadata}
        if is_aoss:
            request['id'] = _id
        else:
            request['_id'] = _id
        requests.append(request)
        return_ids.append(_id)
    bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
    if not is_aoss:
        client.indices.refresh(index=index_name)
    return return_ids