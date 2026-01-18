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
@classmethod
def _from_kwargs(cls: Type[ADBVST], embedding: Embeddings, **kwargs: Any) -> ADBVST:
    known_kwargs = {'collection_name', 'token', 'api_endpoint', 'astra_db_client', 'async_astra_db_client', 'namespace', 'metric', 'batch_size', 'bulk_insert_batch_concurrency', 'bulk_insert_overwrite_concurrency', 'bulk_delete_concurrency', 'batch_concurrency', 'overwrite_concurrency'}
    if kwargs:
        unknown_kwargs = set(kwargs.keys()) - known_kwargs
        if unknown_kwargs:
            warnings.warn(f"Method 'from_texts' of AstraDB vector store invoked with unsupported arguments ({', '.join(sorted(unknown_kwargs))}), which will be ignored.")
    collection_name: str = kwargs['collection_name']
    token = kwargs.get('token')
    api_endpoint = kwargs.get('api_endpoint')
    astra_db_client = kwargs.get('astra_db_client')
    async_astra_db_client = kwargs.get('async_astra_db_client')
    namespace = kwargs.get('namespace')
    metric = kwargs.get('metric')
    return cls(embedding=embedding, collection_name=collection_name, token=token, api_endpoint=api_endpoint, astra_db_client=astra_db_client, async_astra_db_client=async_astra_db_client, namespace=namespace, metric=metric, batch_size=kwargs.get('batch_size'), bulk_insert_batch_concurrency=kwargs.get('bulk_insert_batch_concurrency'), bulk_insert_overwrite_concurrency=kwargs.get('bulk_insert_overwrite_concurrency'), bulk_delete_concurrency=kwargs.get('bulk_delete_concurrency'))