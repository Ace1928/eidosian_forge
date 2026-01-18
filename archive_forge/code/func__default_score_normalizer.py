from __future__ import annotations
import datetime
import os
from typing import (
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _default_score_normalizer(val: float) -> float:
    return 1 - 1 / (1 + np.exp(val))