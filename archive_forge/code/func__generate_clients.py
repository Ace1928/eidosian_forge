from __future__ import annotations
import functools
import uuid
import warnings
from itertools import islice
from operator import itemgetter
from typing import (
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _generate_clients(location: Optional[str]=None, url: Optional[str]=None, port: Optional[int]=6333, grpc_port: int=6334, prefer_grpc: bool=False, https: Optional[bool]=None, api_key: Optional[str]=None, prefix: Optional[str]=None, timeout: Optional[float]=None, host: Optional[str]=None, path: Optional[str]=None, **kwargs: Any) -> Tuple[Any, Any]:
    from qdrant_client import AsyncQdrantClient, QdrantClient
    sync_client = QdrantClient(location=location, url=url, port=port, grpc_port=grpc_port, prefer_grpc=prefer_grpc, https=https, api_key=api_key, prefix=prefix, timeout=timeout, host=host, path=path, **kwargs)
    if location == ':memory:' or path is not None:
        async_client = None
    else:
        async_client = AsyncQdrantClient(location=location, url=url, port=port, grpc_port=grpc_port, prefer_grpc=prefer_grpc, https=https, api_key=api_key, prefix=prefix, timeout=timeout, host=host, path=path, **kwargs)
    return (sync_client, async_client)