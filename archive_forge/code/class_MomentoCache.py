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
class MomentoCache(BaseCache):
    """Cache that uses Momento as a backend. See https://gomomento.com/"""

    def __init__(self, cache_client: momento.CacheClient, cache_name: str, *, ttl: Optional[timedelta]=None, ensure_cache_exists: bool=True):
        """Instantiate a prompt cache using Momento as a backend.

        Note: to instantiate the cache client passed to MomentoCache,
        you must have a Momento account. See https://gomomento.com/.

        Args:
            cache_client (CacheClient): The Momento cache client.
            cache_name (str): The name of the cache to use to store the data.
            ttl (Optional[timedelta], optional): The time to live for the cache items.
                Defaults to None, ie use the client default TTL.
            ensure_cache_exists (bool, optional): Create the cache if it doesn't
                exist. Defaults to True.

        Raises:
            ImportError: Momento python package is not installed.
            TypeError: cache_client is not of type momento.CacheClientObject
            ValueError: ttl is non-null and non-negative
        """
        try:
            from momento import CacheClient
        except ImportError:
            raise ImportError('Could not import momento python package. Please install it with `pip install momento`.')
        if not isinstance(cache_client, CacheClient):
            raise TypeError('cache_client must be a momento.CacheClient object.')
        _validate_ttl(ttl)
        if ensure_cache_exists:
            _ensure_cache_exists(cache_client, cache_name)
        self.cache_client = cache_client
        self.cache_name = cache_name
        self.ttl = ttl

    @classmethod
    def from_client_params(cls, cache_name: str, ttl: timedelta, *, configuration: Optional[momento.config.Configuration]=None, api_key: Optional[str]=None, auth_token: Optional[str]=None, **kwargs: Any) -> MomentoCache:
        """Construct cache from CacheClient parameters."""
        try:
            from momento import CacheClient, Configurations, CredentialProvider
        except ImportError:
            raise ImportError('Could not import momento python package. Please install it with `pip install momento`.')
        if configuration is None:
            configuration = Configurations.Laptop.v1()
        try:
            api_key = auth_token or get_from_env('auth_token', 'MOMENTO_AUTH_TOKEN')
        except ValueError:
            api_key = api_key or get_from_env('api_key', 'MOMENTO_API_KEY')
        credentials = CredentialProvider.from_string(api_key)
        cache_client = CacheClient(configuration, credentials, default_ttl=ttl)
        return cls(cache_client, cache_name, ttl=ttl, **kwargs)

    def __key(self, prompt: str, llm_string: str) -> str:
        """Compute cache key from prompt and associated model and settings.

        Args:
            prompt (str): The prompt run through the language model.
            llm_string (str): The language model version and settings.

        Returns:
            str: The cache key.
        """
        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Lookup llm generations in cache by prompt and associated model and settings.

        Args:
            prompt (str): The prompt run through the language model.
            llm_string (str): The language model version and settings.

        Raises:
            SdkException: Momento service or network error

        Returns:
            Optional[RETURN_VAL_TYPE]: A list of language model generations.
        """
        from momento.responses import CacheGet
        generations: RETURN_VAL_TYPE = []
        get_response = self.cache_client.get(self.cache_name, self.__key(prompt, llm_string))
        if isinstance(get_response, CacheGet.Hit):
            value = get_response.value_string
            generations = _load_generations_from_json(value)
        elif isinstance(get_response, CacheGet.Miss):
            pass
        elif isinstance(get_response, CacheGet.Error):
            raise get_response.inner_exception
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Store llm generations in cache.

        Args:
            prompt (str): The prompt run through the language model.
            llm_string (str): The language model string.
            return_val (RETURN_VAL_TYPE): A list of language model generations.

        Raises:
            SdkException: Momento service or network error
            Exception: Unexpected response
        """
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(f'Momento only supports caching of normal LLM generations, got {type(gen)}')
        key = self.__key(prompt, llm_string)
        value = _dump_generations_to_json(return_val)
        set_response = self.cache_client.set(self.cache_name, key, value, self.ttl)
        from momento.responses import CacheSet
        if isinstance(set_response, CacheSet.Success):
            pass
        elif isinstance(set_response, CacheSet.Error):
            raise set_response.inner_exception
        else:
            raise Exception(f'Unexpected response: {set_response}')

    def clear(self, **kwargs: Any) -> None:
        """Clear the cache.

        Raises:
            SdkException: Momento service or network error
        """
        from momento.responses import CacheFlush
        flush_response = self.cache_client.flush_cache(self.cache_name)
        if isinstance(flush_response, CacheFlush.Success):
            pass
        elif isinstance(flush_response, CacheFlush.Error):
            raise flush_response.inner_exception