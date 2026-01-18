from __future__ import annotations
import hashlib
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC
from datetime import timedelta
from enum import Enum
from functools import lru_cache, wraps
from typing import (
from sqlalchemy import Column, Integer, String, create_engine, delete, select
from sqlalchemy.engine import Row
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from langchain_community.vectorstores.azure_cosmos_db import (
from langchain_core._api.deprecation import deprecated
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM, aget_prompts, get_prompts
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils import get_from_env
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from langchain_community.vectorstores.redis import Redis as RedisVectorstore
class SQLAlchemyCache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(self, engine: Engine, cache_schema: Type[FullLLMCache]=FullLLMCache):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        stmt = select(self.cache_schema.response).where(self.cache_schema.prompt == prompt).where(self.cache_schema.llm == llm_string).order_by(self.cache_schema.idx)
        with Session(self.engine) as session:
            rows = session.execute(stmt).fetchall()
            if rows:
                try:
                    return [loads(row[0]) for row in rows]
                except Exception:
                    logger.warning('Retrieving a cache value that could not be deserialized properly. This is likely due to the cache being in an older format. Please recreate your cache to avoid this error.')
                    return [Generation(text=row[0]) for row in rows]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        items = [self.cache_schema(prompt=prompt, llm=llm_string, response=dumps(gen), idx=i) for i, gen in enumerate(return_val)]
        with Session(self.engine) as session, session.begin():
            for item in items:
                session.merge(item)

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        with Session(self.engine) as session:
            session.query(self.cache_schema).delete()
            session.commit()