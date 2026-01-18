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
class CassandraCache(BaseCache):
    """
    Cache that uses Cassandra / Astra DB as a backend.

    It uses a single Cassandra table.
    The lookup keys (which get to form the primary key) are:
        - prompt, a string
        - llm_string, a deterministic str representation of the model parameters.
          (needed to prevent collisions same-prompt-different-model collisions)
    """

    def __init__(self, session: Optional[CassandraSession]=None, keyspace: Optional[str]=None, table_name: str=CASSANDRA_CACHE_DEFAULT_TABLE_NAME, ttl_seconds: Optional[int]=CASSANDRA_CACHE_DEFAULT_TTL_SECONDS, skip_provisioning: bool=False):
        """
        Initialize with a ready session and a keyspace name.
        Args:
            session (cassandra.cluster.Session): an open Cassandra session
            keyspace (str): the keyspace to use for storing the cache
            table_name (str): name of the Cassandra table to use as cache
            ttl_seconds (optional int): time-to-live for cache entries
                (default: None, i.e. forever)
        """
        try:
            from cassio.table import ElasticCassandraTable
        except (ImportError, ModuleNotFoundError):
            raise ValueError('Could not import cassio python package. Please install it with `pip install cassio`.')
        self.session = session
        self.keyspace = keyspace
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self.kv_cache = ElasticCassandraTable(session=self.session, keyspace=self.keyspace, table=self.table_name, keys=['llm_string', 'prompt'], primary_key_type=['TEXT', 'TEXT'], ttl_seconds=self.ttl_seconds, skip_provisioning=skip_provisioning)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        item = self.kv_cache.get(llm_string=_hash(llm_string), prompt=_hash(prompt))
        if item is not None:
            generations = _loads_generations(item['body_blob'])
            if generations is not None:
                return generations
            else:
                return None
        else:
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        blob = _dumps_generations(return_val)
        self.kv_cache.put(llm_string=_hash(llm_string), prompt=_hash(prompt), body_blob=blob)

    def delete_through_llm(self, prompt: str, llm: LLM, stop: Optional[List[str]]=None) -> None:
        """
        A wrapper around `delete` with the LLM being passed.
        In case the llm(prompt) calls have a `stop` param, you should pass it here
        """
        llm_string = get_prompts({**llm.dict(), **{'stop': stop}}, [])[1]
        return self.delete(prompt, llm_string=llm_string)

    def delete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        return self.kv_cache.delete(llm_string=_hash(llm_string), prompt=_hash(prompt))

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. This is for all LLMs at once."""
        self.kv_cache.clear()