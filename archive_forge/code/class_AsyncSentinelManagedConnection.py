import asyncio
import random
import weakref
from typing import AsyncIterator, Iterable, Mapping, Sequence, Tuple, Type
from aiokeydb.v1.asyncio.core import AsyncKeyDB
from aiokeydb.v1.asyncio.connection import (
from aiokeydb.v1.commands import AsyncSentinelCommands
from aiokeydb.v1.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from aiokeydb.v1.utils import str_if_bytes
class AsyncSentinelManagedConnection(AsyncConnection):

    def __init__(self, **kwargs):
        self.connection_pool = kwargs.pop('connection_pool')
        super().__init__(**kwargs)

    def __repr__(self):
        pool = self.connection_pool
        s = f'{self.__class__.__name__}<service={pool.service_name}'
        if self.host:
            host_info = f',host={self.host},port={self.port}'
            s += host_info
        return s + '>'

    async def connect_to(self, address):
        self.host, self.port = address
        await super().connect()
        if self.connection_pool.check_connection:
            await self.send_command('PING')
            if str_if_bytes(await self.read_response()) != 'PONG':
                raise ConnectionError('PING failed')

    async def _connect_retry(self):
        if self._reader:
            return
        if self.connection_pool.is_master:
            await self.connect_to(await self.connection_pool.get_master_address())
        else:
            async for slave in self.connection_pool.rotate_slaves():
                try:
                    return await self.connect_to(slave)
                except ConnectionError:
                    continue
            raise SlaveNotFoundError

    async def connect(self):
        return await self.retry.call_with_retry(self._connect_retry, lambda error: asyncio.sleep(0))

    async def read_response(self, disable_decoding: bool=False):
        try:
            return await super().read_response(disable_decoding=disable_decoding)
        except ReadOnlyError:
            if self.connection_pool.is_master:
                await self.disconnect()
                raise ConnectionError('The previous master is now a slave')
            raise