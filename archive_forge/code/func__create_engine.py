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
def _create_engine(self) -> sqlalchemy.engine.Engine:
    return sqlalchemy.create_engine(url=self.connection_string, **self.engine_args)