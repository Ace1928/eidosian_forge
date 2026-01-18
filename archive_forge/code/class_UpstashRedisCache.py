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
class UpstashRedisCache(BaseCache):
    """Cache that uses Upstash Redis as a backend."""

    def __init__(self, redis_: Any, *, ttl: Optional[int]=None):
        """
        Initialize an instance of UpstashRedisCache.

        This method initializes an object with Upstash Redis caching capabilities.
        It takes a `redis_` parameter, which should be an instance of an Upstash Redis
        client class, allowing the object to interact with Upstash Redis
        server for caching purposes.

        Parameters:
            redis_: An instance of Upstash Redis client class
                (e.g., Redis) used for caching.
                This allows the object to communicate with
                Redis server for caching operations on.
            ttl (int, optional): Time-to-live (TTL) for cached items in seconds.
                If provided, it sets the time duration for how long cached
                items will remain valid. If not provided, cached items will not
                have an automatic expiration.
        """
        try:
            from upstash_redis import Redis
        except ImportError:
            raise ValueError('Could not import upstash_redis python package. Please install it with `pip install upstash_redis`.')
        if not isinstance(redis_, Redis):
            raise ValueError('Please pass in Upstash Redis object.')
        self.redis = redis_
        self.ttl = ttl

    def _key(self, prompt: str, llm_string: str) -> str:
        """Compute key from prompt and llm_string"""
        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        generations = []
        results = self.redis.hgetall(self._key(prompt, llm_string))
        if results:
            for _, text in results.items():
                generations.append(Generation(text=text))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(f'UpstashRedisCache supports caching of normal LLM generations, got {type(gen)}')
            if isinstance(gen, ChatGeneration):
                warnings.warn('NOTE: Generation has not been cached. UpstashRedisCache does not support caching ChatModel outputs.')
                return
        key = self._key(prompt, llm_string)
        mapping = {str(idx): generation.text for idx, generation in enumerate(return_val)}
        self.redis.hset(key=key, values=mapping)
        if self.ttl is not None:
            self.redis.expire(key, self.ttl)

    def clear(self, **kwargs: Any) -> None:
        """
        Clear cache. If `asynchronous` is True, flush asynchronously.
        This flushes the *whole* db.
        """
        asynchronous = kwargs.get('asynchronous', False)
        if asynchronous:
            asynchronous = 'ASYNC'
        else:
            asynchronous = 'SYNC'
        self.redis.flushdb(flush_type=asynchronous)