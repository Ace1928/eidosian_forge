from __future__ import annotations
import contextlib
import enum
import logging
import uuid
from typing import (
import numpy as np
import sqlalchemy
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session
from sqlalchemy.sql import quoted_name
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def _get_operator(self) -> str:
    if self.distance_strategy == DistanceStrategy.COSINE:
        return '<=>'
    elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
        return '<->'
    elif self.distance_strategy == DistanceStrategy.HAMMING:
        return '<+>'
    else:
        raise ValueError(f'No supported operator for distance_strategy of {self._distance_strategy}.')