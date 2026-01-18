from __future__ import annotations
import asyncio
import inspect
from asyncio import InvalidStateError, Task
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Optional, Union
class _AstraDBCollectionEnvironment(_AstraDBEnvironment):

    def __init__(self, collection_name: str, token: Optional[str]=None, api_endpoint: Optional[str]=None, astra_db_client: Optional[AstraDB]=None, async_astra_db_client: Optional[AsyncAstraDB]=None, namespace: Optional[str]=None, setup_mode: SetupMode=SetupMode.SYNC, pre_delete_collection: bool=False, embedding_dimension: Union[int, Awaitable[int], None]=None, metric: Optional[str]=None) -> None:
        from astrapy.db import AstraDBCollection, AsyncAstraDBCollection
        super().__init__(token, api_endpoint, astra_db_client, async_astra_db_client, namespace)
        self.collection_name = collection_name
        self.collection = AstraDBCollection(collection_name=collection_name, astra_db=self.astra_db)
        self.async_collection = AsyncAstraDBCollection(collection_name=collection_name, astra_db=self.async_astra_db)
        self.async_setup_db_task: Optional[Task] = None
        if setup_mode == SetupMode.ASYNC:
            async_astra_db = self.async_astra_db

            async def _setup_db() -> None:
                if pre_delete_collection:
                    await async_astra_db.delete_collection(collection_name)
                if inspect.isawaitable(embedding_dimension):
                    dimension = await embedding_dimension
                else:
                    dimension = embedding_dimension
                await async_astra_db.create_collection(collection_name, dimension=dimension, metric=metric)
            self.async_setup_db_task = asyncio.create_task(_setup_db())
        elif setup_mode == SetupMode.SYNC:
            if pre_delete_collection:
                self.astra_db.delete_collection(collection_name)
            if inspect.isawaitable(embedding_dimension):
                raise ValueError('Cannot use an awaitable embedding_dimension with async_setup set to False')
            self.astra_db.create_collection(collection_name, dimension=embedding_dimension, metric=metric)

    def ensure_db_setup(self) -> None:
        if self.async_setup_db_task:
            try:
                self.async_setup_db_task.result()
            except InvalidStateError:
                raise ValueError("Asynchronous setup of the DB not finished. NB: AstraDB components sync methods shouldn't be called from the event loop. Consider using their async equivalents.")

    async def aensure_db_setup(self) -> None:
        if self.async_setup_db_task:
            await self.async_setup_db_task