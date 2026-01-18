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
def _bytes2embedding(blob: bytes) -> Any:
    emb = np.frombuffer(blob, dtype='float32')
    return emb