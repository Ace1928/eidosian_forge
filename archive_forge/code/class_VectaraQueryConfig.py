from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
@dataclass
class VectaraQueryConfig:
    """Configuration for Vectara query.

    k: Number of Documents to return. Defaults to 10.
    lambda_val: lexical match parameter for hybrid search.
    filter Dictionary of argument(s) to filter on metadata. For example a
        filter can be "doc.rating > 3.0 and part.lang = 'deu'"} see
        https://docs.vectara.com/docs/search-apis/sql/filter-overview
        for more details.
    score_threshold: minimal score threshold for the result.
        If defined, results with score less than this value will be
        filtered out.
    n_sentence_context: number of sentences before/after the matching segment
        to add, defaults to 2
    mmr_config: MMRConfig configuration dataclass
    summary_config: SummaryConfig configuration dataclass
    """
    k: int = 10
    lambda_val: float = 0.0
    filter: str = ''
    score_threshold: Optional[float] = None
    n_sentence_context: int = 2
    mmr_config: MMRConfig = field(default_factory=MMRConfig)
    summary_config: SummaryConfig = field(default_factory=SummaryConfig)