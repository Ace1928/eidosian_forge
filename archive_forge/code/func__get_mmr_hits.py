from __future__ import annotations
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _get_mmr_hits(embedding: List[float], k: int, lambda_mult: float, prefetch_hits: List[DocDict]) -> List[Document]:
    mmr_chosen_indices = maximal_marginal_relevance(np.array(embedding, dtype=np.float32), [prefetch_hit['$vector'] for prefetch_hit in prefetch_hits], k=k, lambda_mult=lambda_mult)
    mmr_hits = [prefetch_hit for prefetch_index, prefetch_hit in enumerate(prefetch_hits) if prefetch_index in mmr_chosen_indices]
    return [Document(page_content=hit['content'], metadata=hit['metadata']) for hit in mmr_hits]