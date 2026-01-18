import sys
import time
import anyio
import typing
import logging
import asyncio
import functools
import contextlib
from pydantic import BaseModel
from pydantic.types import ByteSize
from aiokeydb.v2.typing import Number, KeyT, ExpiryT, AbsExpiryT, PatternT
from aiokeydb.v2.lock import Lock, AsyncLock
from aiokeydb.v2.core import KeyDB, PubSub, Pipeline, PipelineT, PubSubT
from aiokeydb.v2.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline, AsyncPipelineT, AsyncPubSubT
from aiokeydb.v2.connection import Encoder, ConnectionPool, AsyncConnectionPool
from aiokeydb.v2.exceptions import (
from aiokeydb.v2.types import KeyDBUri, ENOVAL
from aiokeydb.v2.configs import KeyDBSettings, settings as default_settings
from aiokeydb.v2.utils import full_name, args_to_key
from aiokeydb.v2.utils.helpers import create_retryable_client
from aiokeydb.v2.serializers import BaseSerializer
from inspect import iscoroutinefunction
class ClientPools(BaseModel):
    """
    Holds the reference for connection pools
    """
    name: str
    pool: typing.Union[ConnectionPool, typing.Type[ConnectionPool]]
    apool: typing.Union[AsyncConnectionPool, typing.Type[AsyncConnectionPool]]

    class Config:
        arbitrary_types_allowed = True

    def with_db_id(self, db_id: int) -> 'ClientPools':
        """
        Returns a new ClientPools with the given db_id
        """
        return ClientPools(name=self.name, pool=self.pool.with_db_id(db_id), apool=self.apool.with_db_id(db_id))