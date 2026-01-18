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
from aiokeydb.typing import Number, KeyT, ExpiryT, AbsExpiryT, PatternT
from aiokeydb.lock import Lock, AsyncLock
from aiokeydb.core import KeyDB, PubSub, Pipeline, PipelineT, PubSubT
from aiokeydb.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline, AsyncPipelineT, AsyncPubSubT
from aiokeydb.connection import Encoder, ConnectionPool, AsyncConnectionPool
from aiokeydb.exceptions import (
from aiokeydb.types import KeyDBUri, ENOVAL
from aiokeydb.configs import KeyDBSettings
from aiokeydb.utils import full_name, args_to_key, get_keydb_settings
from aiokeydb.utils.helpers import create_retryable_client, afail_after
from aiokeydb.utils.logs import logger
from .cachify import cachify, create_cachify, FT
from aiokeydb.serializers import BaseSerializer
from inspect import iscoroutinefunction
def _configure_dict_methods(session: 'KeyDBSession', method: typing.Optional[str]=None, async_enabled: typing.Optional[bool]=None):
    """
    Configures the Dict get/set methods
    """
    if method is None:
        method = session.state.dict_method
    if async_enabled is None:
        async_enabled = session.state.dict_async_enabled
    if method == session.state.dict_method and async_enabled == session.state.dict_async_enabled:
        return
    if async_enabled:

        async def getitem(self: 'KeyDBSession', key: KeyT) -> typing.Any:
            if method == 'hash':
                value = await self.async_client.hget(self.dict_hash_key, key)
            else:
                value = await self.async_client.get(key)
            if value is None:
                key_value = f'{self.dict_hash_key}:{key}' if method == 'hash' else key
                raise KeyError(key_value)
            if self.dict_decoder is not False:
                value = self.dict_decoder(value)
            return value
    else:

        def getitem(self: 'KeyDBSession', key: KeyT) -> typing.Any:
            if method == 'hash':
                value = self.client.hget(self.dict_hash_key, key)
            else:
                value = self.client.get(key)
            if value is None:
                key_value = f'{self.dict_hash_key}:{key}' if method == 'hash' else key
                raise KeyError(key_value)
            if self.dict_decoder is not False:
                value = self.dict_decoder(value)
            return value
    setattr(KeyDBSession, '__getitem__', getitem)
    session.__getitem__ = getitem
    session.state.dict_method = method
    session.state.dict_async_enabled = async_enabled