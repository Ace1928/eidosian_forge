from abc import ABC
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class DocArrayDoc(BaseDoc):
    text: Optional[str] = Field(default=None, required=False)
    embedding: Optional[NdArray] = Field(**embeddings_params)
    metadata: Optional[dict] = Field(default=None, required=False)