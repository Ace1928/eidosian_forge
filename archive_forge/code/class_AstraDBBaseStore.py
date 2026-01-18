from __future__ import annotations
import base64
from abc import ABC, abstractmethod
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.stores import BaseStore, ByteStore
from langchain_community.utilities.astradb import (
class AstraDBBaseStore(Generic[V], BaseStore[str, V], ABC):
    """Base class for the DataStax AstraDB data store."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.astra_env = _AstraDBCollectionEnvironment(*args, **kwargs)
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    @abstractmethod
    def decode_value(self, value: Any) -> Optional[V]:
        """Decodes value from Astra DB"""

    @abstractmethod
    def encode_value(self, value: Optional[V]) -> Any:
        """Encodes value for Astra DB"""

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        self.astra_env.ensure_db_setup()
        docs_dict = {}
        for doc in self.collection.paginated_find(filter={'_id': {'$in': list(keys)}}):
            docs_dict[doc['_id']] = doc.get('value')
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    async def amget(self, keys: Sequence[str]) -> List[Optional[V]]:
        await self.astra_env.aensure_db_setup()
        docs_dict = {}
        async for doc in self.async_collection.paginated_find(filter={'_id': {'$in': list(keys)}}):
            docs_dict[doc['_id']] = doc.get('value')
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        self.astra_env.ensure_db_setup()
        for k, v in key_value_pairs:
            self.collection.upsert({'_id': k, 'value': self.encode_value(v)})

    async def amset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        await self.astra_env.aensure_db_setup()
        for k, v in key_value_pairs:
            await self.async_collection.upsert({'_id': k, 'value': self.encode_value(v)})

    def mdelete(self, keys: Sequence[str]) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.delete_many(filter={'_id': {'$in': list(keys)}})

    async def amdelete(self, keys: Sequence[str]) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many(filter={'_id': {'$in': list(keys)}})

    def yield_keys(self, *, prefix: Optional[str]=None) -> Iterator[str]:
        self.astra_env.ensure_db_setup()
        docs = self.collection.paginated_find()
        for doc in docs:
            key = doc['_id']
            if not prefix or key.startswith(prefix):
                yield key

    async def ayield_keys(self, *, prefix: Optional[str]=None) -> AsyncIterator[str]:
        await self.astra_env.aensure_db_setup()
        async for doc in self.async_collection.paginated_find():
            key = doc['_id']
            if not prefix or key.startswith(prefix):
                yield key