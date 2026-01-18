from __future__ import annotations
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class IndexParams:
    """Tencent vector DB Index params.

    See the following documentation for details:
    https://cloud.tencent.com/document/product/1709/95826
    """

    def __init__(self, dimension: int, shard: int=1, replicas: int=2, index_type: str='HNSW', metric_type: str='L2', params: Optional[Dict]=None):
        self.dimension = dimension
        self.shard = shard
        self.replicas = replicas
        self.index_type = index_type
        self.metric_type = metric_type
        self.params = params