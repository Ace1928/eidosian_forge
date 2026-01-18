from __future__ import annotations
import logging
import math
import warnings
from abc import ABC, abstractmethod
from typing import (
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
def _get_retriever_tags(self) -> List[str]:
    """Get tags for retriever."""
    tags = [self.__class__.__name__]
    if self.embeddings:
        tags.append(self.embeddings.__class__.__name__)
    return tags