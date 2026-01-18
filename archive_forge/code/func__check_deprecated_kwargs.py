from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _check_deprecated_kwargs(self, kwargs: Mapping[str, Any]) -> None:
    """Check for deprecated kwargs."""
    deprecated_kwargs = {'redis_host': 'redis_url', 'redis_port': 'redis_url', 'redis_password': 'redis_url', 'content_key': 'index_schema', 'vector_key': 'vector_schema', 'distance_metric': 'vector_schema'}
    for key, value in kwargs.items():
        if key in deprecated_kwargs:
            raise ValueError(f"Keyword argument '{key}' is deprecated. Please use '{deprecated_kwargs[key]}' instead.")