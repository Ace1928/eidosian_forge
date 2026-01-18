import asyncio
import random
import weakref
from typing import AsyncIterator, Iterable, Mapping, Optional, Sequence, Tuple, Type
from redis.asyncio.client import Redis
from redis.asyncio.connection import (
from redis.commands import AsyncSentinelCommands
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
class SentinelConnectionPool(ConnectionPool):
    """
    Sentinel backed connection pool.

    If ``check_connection`` flag is set to True, SentinelManagedConnection
    sends a PING command right after establishing the connection.
    """

    def __init__(self, service_name, sentinel_manager, **kwargs):
        kwargs['connection_class'] = kwargs.get('connection_class', SentinelManagedSSLConnection if kwargs.pop('ssl', False) else SentinelManagedConnection)
        self.is_master = kwargs.pop('is_master', True)
        self.check_connection = kwargs.pop('check_connection', False)
        super().__init__(**kwargs)
        self.connection_kwargs['connection_pool'] = weakref.proxy(self)
        self.service_name = service_name
        self.sentinel_manager = sentinel_manager
        self.master_address = None
        self.slave_rr_counter = None

    def __repr__(self):
        return f'{self.__class__.__name__}<service={self.service_name}({self.is_master and 'master' or 'slave'})>'

    def reset(self):
        super().reset()
        self.master_address = None
        self.slave_rr_counter = None

    def owns_connection(self, connection: Connection):
        check = not self.is_master or (self.is_master and self.master_address == (connection.host, connection.port))
        return check and super().owns_connection(connection)

    async def get_master_address(self):
        master_address = await self.sentinel_manager.discover_master(self.service_name)
        if self.is_master:
            if self.master_address != master_address:
                self.master_address = master_address
                await self.disconnect(inuse_connections=False)
        return master_address

    async def rotate_slaves(self) -> AsyncIterator:
        """Round-robin slave balancer"""
        slaves = await self.sentinel_manager.discover_slaves(self.service_name)
        if slaves:
            if self.slave_rr_counter is None:
                self.slave_rr_counter = random.randint(0, len(slaves) - 1)
            for _ in range(len(slaves)):
                self.slave_rr_counter = (self.slave_rr_counter + 1) % len(slaves)
                slave = slaves[self.slave_rr_counter]
                yield slave
        try:
            yield (await self.get_master_address())
        except MasterNotFoundError:
            pass
        raise SlaveNotFoundError(f'No slave found for {self.service_name!r}')