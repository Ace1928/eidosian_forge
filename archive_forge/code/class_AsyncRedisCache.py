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
class AsyncRedisCache(_RedisCacheBase):
    """
    Cache that uses Redis as a backend. Allows to use an
    async `redis.asyncio.Redis` client.
    """

    def __init__(self, redis_: Any, *, ttl: Optional[int]=None):
        """
        Initialize an instance of AsyncRedisCache.

        This method initializes an object with Redis caching capabilities.
        It takes a `redis_` parameter, which should be an instance of a Redis
        client class (`redis.asyncio.Redis`), allowing the object
        to interact with a Redis server for caching purposes.

        Parameters:
            redis_ (Any): An instance of a Redis client class
                (`redis.asyncio.Redis`) to be used for caching.
                This allows the object to communicate with a
                Redis server for caching operations.
            ttl (int, optional): Time-to-live (TTL) for cached items in seconds.
                If provided, it sets the time duration for how long cached
                items will remain valid. If not provided, cached items will not
                have an automatic expiration.
        """
        try:
            from redis.asyncio import Redis
        except ImportError:
            raise ValueError('Could not import `redis.asyncio` python package. Please install it with `pip install redis`.')
        if not isinstance(redis_, Redis):
            raise ValueError('Please pass a valid `redis.asyncio.Redis` client.')
        self.redis = redis_
        self.ttl = ttl

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        raise NotImplementedError('This async Redis cache does not implement `lookup()` method. Consider using the async `alookup()` version.')

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string. Async version."""
        try:
            results = await self.redis.hgetall(self._key(prompt, llm_string))
            return self._get_generations(results)
        except Exception as e:
            logger.error(f'Redis async lookup failed: {e}')
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        raise NotImplementedError('This async Redis cache does not implement `update()` method. Consider using the async `aupdate()` version.')

    async def aupdate(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string. Async version."""
        self._ensure_generation_type(return_val)
        key = self._key(prompt, llm_string)
        try:
            async with self.redis.pipeline() as pipe:
                self._configure_pipeline_for_update(key, pipe, return_val, self.ttl)
                await pipe.execute()
        except Exception as e:
            logger.error(f'Redis async update failed: {e}')

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. If `asynchronous` is True, flush asynchronously."""
        raise NotImplementedError('This async Redis cache does not implement `clear()` method. Consider using the async `aclear()` version.')

    async def aclear(self, **kwargs: Any) -> None:
        """
        Clear cache. If `asynchronous` is True, flush asynchronously.
        Async version.
        """
        try:
            asynchronous = kwargs.get('asynchronous', False)
            await self.redis.flushdb(asynchronous=asynchronous, **kwargs)
        except Exception as e:
            logger.error(f'Redis async clear failed: {e}')