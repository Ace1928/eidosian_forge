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
def _build_condition(self, key: str, value: Any) -> List[rest.FieldCondition]:
    from qdrant_client.http import models as rest
    out = []
    if isinstance(value, dict):
        for _key, value in value.items():
            out.extend(self._build_condition(f'{key}.{_key}', value))
    elif isinstance(value, list):
        for _value in value:
            if isinstance(_value, dict):
                out.extend(self._build_condition(f'{key}[]', _value))
            else:
                out.extend(self._build_condition(f'{key}', _value))
    else:
        out.append(rest.FieldCondition(key=f'{self.metadata_payload_key}.{key}', match=rest.MatchValue(value=value)))
    return out