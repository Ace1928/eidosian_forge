from abc import ABC
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _get_doc_cls(**embeddings_params: Any) -> Type['BaseDoc']:
    """Get docarray Document class describing the schema of DocIndex."""
    from docarray import BaseDoc
    from docarray.typing import NdArray

    class DocArrayDoc(BaseDoc):
        text: Optional[str] = Field(default=None, required=False)
        embedding: Optional[NdArray] = Field(**embeddings_params)
        metadata: Optional[dict] = Field(default=None, required=False)
    return DocArrayDoc