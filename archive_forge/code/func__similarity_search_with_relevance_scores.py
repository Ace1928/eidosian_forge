from abc import ABC
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _similarity_search_with_relevance_scores(self, query: str, k: int=4, **kwargs: Any) -> List[Tuple[Document, float]]:
    """Return docs and relevance scores, normalized on a scale from 0 to 1.

        0 is dissimilar, 1 is most similar.
        """
    raise NotImplementedError()