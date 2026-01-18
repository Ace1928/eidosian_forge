from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def add_embeddings(self, texts: Iterable[str], embeddings: List[List[float]], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, **kwargs: Any) -> List[str]:
    """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            ids: List of ids for the text embedding pairs
            kwargs: vectorstore specific parameters
        """
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    if not metadatas:
        metadatas = [{} for _ in texts]
    records = []
    for text, embedding, metadata, id in zip(texts, embeddings, metadatas, ids):
        buf = struct.pack('%sf' % self.dimensions, *embedding)
        records.append([text, buf, json.dumps(metadata), id])
    self.EmbeddingStore.insert_records(records)
    return ids