from __future__ import annotations
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _create_partition_if_not_exists(self, partition: str) -> None:
    """Create a Partition in current Collection."""
    self._collection.create_partition(partition)