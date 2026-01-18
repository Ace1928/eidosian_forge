from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _get_vector_index_ivf(self, kind: str, num_lists: int, similarity: str, dimensions: int) -> Dict[str, Any]:
    command = {'createIndexes': self._collection.name, 'indexes': [{'name': self._index_name, 'key': {self._embedding_key: 'cosmosSearch'}, 'cosmosSearchOptions': {'kind': kind, 'numLists': num_lists, 'similarity': similarity, 'dimensions': dimensions}}]}
    return command