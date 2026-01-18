from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
Returns a list of documents.

        Args:
            embeddings: The query vector
            k: the number of documents to return
            ef_search: Specifies the size of the dynamic candidate list
                that HNSW index uses during search. A higher value of
                efSearch provides better recall at cost of speed.

        Returns:
            A list of documents closest to the query vector
        