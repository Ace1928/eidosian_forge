import random
import weakref
from typing import Optional
from redis.client import Redis
from redis.commands import SentinelCommands
from redis.connection import Connection, ConnectionPool, SSLConnection
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
def _connect_retry(self):
    if self._sock:
        return
    if self.connection_pool.is_master:
        self.connect_to(self.connection_pool.get_master_address())
    else:
        for slave in self.connection_pool.rotate_slaves():
            try:
                return self.connect_to(slave)
            except ConnectionError:
                continue
        raise SlaveNotFoundError