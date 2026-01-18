from __future__ import annotations
import contextlib
import enum
import json
import logging
import uuid
from typing import (
import numpy as np
import sqlalchemy
from langchain_core._api import deprecated, warn_deprecated
from sqlalchemy import SQLColumnExpression, delete, func
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import Session, relationship
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _create_filter_clause_json_deprecated(self, filter: Any) -> List[SQLColumnExpression]:
    """Convert filters from IR to SQL clauses.

        **DEPRECATED** This functionality will be deprecated in the future.

        It implements translation of filters for a schema that uses JSON
        for metadata rather than the JSONB field which is more efficient
        for querying.
        """
    filter_clauses = []
    for key, value in filter.items():
        if isinstance(value, dict):
            filter_by_metadata = self._create_filter_clause_deprecated(key, value)
            if filter_by_metadata is not None:
                filter_clauses.append(filter_by_metadata)
        else:
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext == str(value)
            filter_clauses.append(filter_by_metadata)
    return filter_clauses