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
@classmethod
def _build_payloads(cls, texts: Iterable[str], metadatas: Optional[List[dict]], content_payload_key: str, metadata_payload_key: str) -> List[dict]:
    payloads = []
    for i, text in enumerate(texts):
        if text is None:
            raise ValueError('At least one of the texts is None. Please remove it before calling .from_texts or .add_texts on Qdrant instance.')
        metadata = metadatas[i] if metadatas is not None else None
        payloads.append({content_payload_key: text, metadata_payload_key: metadata})
    return payloads