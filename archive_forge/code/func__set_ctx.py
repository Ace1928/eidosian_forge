from __future__ import annotations
import asyncio
import typing
import logging
from aiokeydb.v1.lock import Lock
from aiokeydb.v1.connection import Encoder, ConnectionPool, BlockingConnectionPool, Connection
from aiokeydb.v1.core import KeyDB, PubSub, Pipeline
from aiokeydb.v1.typing import Number, KeyT, AbsExpiryT, ExpiryT
from aiokeydb.v1.asyncio.lock import AsyncLock
from aiokeydb.v1.asyncio.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline
from aiokeydb.v1.asyncio.connection import AsyncConnectionPool, AsyncBlockingConnectionPool, AsyncConnection
from aiokeydb.v1.client.config import KeyDBSettings
from aiokeydb.v1.client.types import KeyDBUri
from aiokeydb.v1.client.schemas.session import KeyDBSession, ClientPools
from aiokeydb.v1.client.serializers import SerializerType, BaseSerializer
def _set_ctx(cls, session: KeyDBSession, name: typing.Optional[str]=None):
    """
        Sets the current session context
        """
    cls._ctx = session
    cls.current = name or session.name
    logger.log(msg=f'Setting to Current Session: {cls.current}', level=cls.settings.loglevel)