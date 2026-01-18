import random
import weakref
from typing import Optional
from redis.client import Redis
from redis.commands import SentinelCommands
from redis.connection import Connection, ConnectionPool, SSLConnection
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
class SentinelConnectionPoolProxy:

    def __init__(self, connection_pool, is_master, check_connection, service_name, sentinel_manager):
        self.connection_pool_ref = weakref.ref(connection_pool)
        self.is_master = is_master
        self.check_connection = check_connection
        self.service_name = service_name
        self.sentinel_manager = sentinel_manager
        self.reset()

    def reset(self):
        self.master_address = None
        self.slave_rr_counter = None

    def get_master_address(self):
        master_address = self.sentinel_manager.discover_master(self.service_name)
        if self.is_master and self.master_address != master_address:
            self.master_address = master_address
            connection_pool = self.connection_pool_ref()
            if connection_pool is not None:
                connection_pool.disconnect(inuse_connections=False)
        return master_address

    def rotate_slaves(self):
        slaves = self.sentinel_manager.discover_slaves(self.service_name)
        if slaves:
            if self.slave_rr_counter is None:
                self.slave_rr_counter = random.randint(0, len(slaves) - 1)
            for _ in range(len(slaves)):
                self.slave_rr_counter = (self.slave_rr_counter + 1) % len(slaves)
                slave = slaves[self.slave_rr_counter]
                yield slave
        try:
            yield self.get_master_address()
        except MasterNotFoundError:
            pass
        raise SlaveNotFoundError(f'No slave found for {self.service_name!r}')