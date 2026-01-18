from __future__ import annotations
import functools
import uuid
import warnings
from itertools import islice
from operator import itemgetter
from typing import (
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _cosine_relevance_score_fn(distance: float) -> float:
    """Normalize the distance to a score on a scale [0, 1]."""
    return (distance + 1.0) / 2.0