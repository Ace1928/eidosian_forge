from __future__ import annotations
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore

        Delete by IDs.

        Args:
            ids: List of ids to delete.
        