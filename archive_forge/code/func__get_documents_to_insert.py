from __future__ import annotations
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _get_documents_to_insert(texts: Iterable[str], embedding_vectors: List[List[float]], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None) -> List[DocDict]:
    if ids is None:
        ids = [uuid.uuid4().hex for _ in texts]
    if metadatas is None:
        metadatas = [{} for _ in texts]
    documents_to_insert = [{'content': b_txt, '_id': b_id, '$vector': b_emb, 'metadata': b_md} for b_txt, b_emb, b_id, b_md in zip(texts, embedding_vectors, ids, metadatas)]
    uniqued_documents_to_insert = _unique_list(documents_to_insert[::-1], lambda document: document['_id'])[::-1]
    return uniqued_documents_to_insert