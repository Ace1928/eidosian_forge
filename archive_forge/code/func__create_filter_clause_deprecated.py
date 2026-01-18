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
def _create_filter_clause_deprecated(self, key, value):
    """Deprecated functionality.

        This is for backwards compatibility with the JSON based schema for metadata.
        It uses incorrect operator syntax (operators are not prefixed with $).

        This implementation is not efficient, and has bugs associated with
        the way that it handles numeric filter clauses.
        """
    IN, NIN, BETWEEN, GT, LT, NE = ('in', 'nin', 'between', 'gt', 'lt', 'ne')
    EQ, LIKE, CONTAINS, OR, AND = ('eq', 'like', 'contains', 'or', 'and')
    value_case_insensitive = {k.lower(): v for k, v in value.items()}
    if IN in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.in_(value_case_insensitive[IN])
    elif NIN in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.not_in(value_case_insensitive[NIN])
    elif BETWEEN in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.between(str(value_case_insensitive[BETWEEN][0]), str(value_case_insensitive[BETWEEN][1]))
    elif GT in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext > str(value_case_insensitive[GT])
    elif LT in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext < str(value_case_insensitive[LT])
    elif NE in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext != str(value_case_insensitive[NE])
    elif EQ in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext == str(value_case_insensitive[EQ])
    elif LIKE in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.like(value_case_insensitive[LIKE])
    elif CONTAINS in map(str.lower, value):
        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.contains(value_case_insensitive[CONTAINS])
    elif OR in map(str.lower, value):
        or_clauses = [self._create_filter_clause_deprecated(key, sub_value) for sub_value in value_case_insensitive[OR]]
        filter_by_metadata = sqlalchemy.or_(*or_clauses)
    elif AND in map(str.lower, value):
        and_clauses = [self._create_filter_clause_deprecated(key, sub_value) for sub_value in value_case_insensitive[AND]]
        filter_by_metadata = sqlalchemy.and_(*and_clauses)
    else:
        filter_by_metadata = None
    return filter_by_metadata