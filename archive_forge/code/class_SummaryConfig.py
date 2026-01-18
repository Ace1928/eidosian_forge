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
class SummaryConfig:
    """Configuration for summary generation.

    is_enabled: True if summary is enabled, False otherwise
    max_results: maximum number of results to summarize
    response_lang: requested language for the summary
    prompt_name: name of the prompt to use for summarization
      (see https://docs.vectara.com/docs/learn/grounded-generation/select-a-summarizer)
    """
    is_enabled: bool = False
    max_results: int = 7
    response_lang: str = 'eng'
    prompt_name: str = 'vectara-summary-ext-v1.2.0'