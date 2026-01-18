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
@deprecated(since='0.0.28', removal='0.2.0', alternative_import='langchain_astradb.AstraDBCache')
class AstraDBCache(BaseCache):

    @staticmethod
    def _make_id(prompt: str, llm_string: str) -> str:
        return f'{_hash(prompt)}#{_hash(llm_string)}'

    def __init__(self, *, collection_name: str=ASTRA_DB_CACHE_DEFAULT_COLLECTION_NAME, token: Optional[str]=None, api_endpoint: Optional[str]=None, astra_db_client: Optional[AstraDB]=None, async_astra_db_client: Optional[AsyncAstraDB]=None, namespace: Optional[str]=None, pre_delete_collection: bool=False, setup_mode: SetupMode=SetupMode.SYNC):
        """
        Cache that uses Astra DB as a backend.

        It uses a single collection as a kv store
        The lookup keys, combined in the _id of the documents, are:
            - prompt, a string
            - llm_string, a deterministic str representation of the model parameters.
              (needed to prevent same-prompt-different-model collisions)

        Args:
            collection_name: name of the Astra DB collection to create/use.
            token: API token for Astra DB usage.
            api_endpoint: full URL to the API endpoint,
                such as `https://<DB-ID>-us-east1.apps.astra.datastax.com`.
            astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AstraDB' instance.
            async_astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance.
            namespace: namespace (aka keyspace) where the
                collection is created. Defaults to the database's "default namespace".
            setup_mode: mode used to create the Astra DB collection (SYNC, ASYNC or
                OFF).
            pre_delete_collection: whether to delete the collection
                before creating it. If False and the collection already exists,
                the collection will be used as is.
        """
        self.astra_env = _AstraDBCollectionEnvironment(collection_name=collection_name, token=token, api_endpoint=api_endpoint, astra_db_client=astra_db_client, async_astra_db_client=async_astra_db_client, namespace=namespace, setup_mode=setup_mode, pre_delete_collection=pre_delete_collection)
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        item = self.collection.find_one(filter={'_id': doc_id}, projection={'body_blob': 1})['data']['document']
        return _loads_generations(item['body_blob']) if item is not None else None

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        item = (await self.async_collection.find_one(filter={'_id': doc_id}, projection={'body_blob': 1}))['data']['document']
        return _loads_generations(item['body_blob']) if item is not None else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        blob = _dumps_generations(return_val)
        self.collection.upsert({'_id': doc_id, 'body_blob': blob})

    async def aupdate(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        blob = _dumps_generations(return_val)
        await self.async_collection.upsert({'_id': doc_id, 'body_blob': blob})

    def delete_through_llm(self, prompt: str, llm: LLM, stop: Optional[List[str]]=None) -> None:
        """
        A wrapper around `delete` with the LLM being passed.
        In case the llm(prompt) calls have a `stop` param, you should pass it here
        """
        llm_string = get_prompts({**llm.dict(), **{'stop': stop}}, [])[1]
        return self.delete(prompt, llm_string=llm_string)

    async def adelete_through_llm(self, prompt: str, llm: LLM, stop: Optional[List[str]]=None) -> None:
        """
        A wrapper around `adelete` with the LLM being passed.
        In case the llm(prompt) calls have a `stop` param, you should pass it here
        """
        llm_string = (await aget_prompts({**llm.dict(), **{'stop': stop}}, []))[1]
        return await self.adelete(prompt, llm_string=llm_string)

    def delete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        self.collection.delete_one(doc_id)

    async def adelete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        await self.async_collection.delete_one(doc_id)

    def clear(self, **kwargs: Any) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.clear()

    async def aclear(self, **kwargs: Any) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.clear()