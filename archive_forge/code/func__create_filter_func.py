from __future__ import annotations
import logging
import operator
import os
import pickle
import uuid
import warnings
from pathlib import Path
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import (
@staticmethod
def _create_filter_func(filter: Optional[Union[Callable, Dict[str, Any]]]) -> Callable[[Dict[str, Any]], bool]:
    """
        Create a filter function based on the provided filter.

        Args:
            filter: A callable or a dictionary representing the filter
            conditions for documents.

        Returns:
            Callable[[Dict[str, Any]], bool]: A function that takes Document's metadata
            and returns True if it satisfies the filter conditions, otherwise False.
        """
    if callable(filter):
        return filter
    if not isinstance(filter, dict):
        raise ValueError(f'filter must be a dict of metadata or a callable, not {type(filter)}')

    def filter_func(metadata: Dict[str, Any]) -> bool:
        return all((metadata.get(key) in value if isinstance(value, list) else metadata.get(key) == value for key, value in filter.items()))
    return filter_func