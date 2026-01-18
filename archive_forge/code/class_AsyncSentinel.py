import asyncio
import random
import weakref
from typing import AsyncIterator, Iterable, Mapping, Sequence, Tuple, Type
from aiokeydb.v1.asyncio.core import AsyncKeyDB
from aiokeydb.v1.asyncio.connection import (
from aiokeydb.v1.commands import AsyncSentinelCommands
from aiokeydb.v1.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from aiokeydb.v1.utils import str_if_bytes
class AsyncSentinel(AsyncSentinelCommands):
    """
    AsyncKeyDB Sentinel cluster client

    >>> from redis.sentinel import Sentinel
    >>> sentinel = Sentinel([('localhost', 26379)], socket_timeout=0.1)
    >>> master = sentinel.master_for('mymaster', socket_timeout=0.1)
    >>> await master.set('foo', 'bar')
    >>> slave = sentinel.slave_for('mymaster', socket_timeout=0.1)
    >>> await slave.get('foo')
    b'bar'

    ``sentinels`` is a list of sentinel nodes. Each node is represented by
    a pair (hostname, port).

    ``min_other_sentinels`` defined a minimum number of peers for a sentinel.
    When querying a sentinel, if it doesn't meet this threshold, responses
    from that sentinel won't be considered valid.

    ``sentinel_kwargs`` is a dictionary of connection arguments used when
    connecting to sentinel instances. Any argument that can be passed to
    a normal AsyncKeyDB connection can be specified here. If ``sentinel_kwargs`` is
    not specified, any socket_timeout and socket_keepalive options specified
    in ``connection_kwargs`` will be used.

    ``connection_kwargs`` are keyword arguments that will be used when
    establishing a connection to a AsyncKeyDB server.
    """

    def __init__(self, sentinels, min_other_sentinels=0, sentinel_kwargs=None, **connection_kwargs):
        if sentinel_kwargs is None:
            sentinel_kwargs = {k: v for k, v in connection_kwargs.items() if k.startswith('socket_')}
        self.sentinel_kwargs = sentinel_kwargs
        self.sentinels = [AsyncKeyDB(host=hostname, port=port, **self.sentinel_kwargs) for hostname, port in sentinels]
        self.min_other_sentinels = min_other_sentinels
        self.connection_kwargs = connection_kwargs

    async def execute_command(self, *args, **kwargs):
        """
        Execute Sentinel command in sentinel nodes.
        once - If set to True, then execute the resulting command on a single
               node at random, rather than across the entire sentinel cluster.
        """
        once = bool(kwargs.get('once', False))
        if 'once' in kwargs.keys():
            kwargs.pop('once')
        if once:
            tasks = [asyncio.Task(sentinel.execute_command(*args, **kwargs)) for sentinel in self.sentinels]
            await asyncio.gather(*tasks)
        else:
            await random.choice(self.sentinels).execute_command(*args, **kwargs)
        return True

    def __repr__(self):
        sentinel_addresses = []
        for sentinel in self.sentinels:
            sentinel_addresses.append(f'{sentinel.connection_pool.connection_kwargs['host']}:{sentinel.connection_pool.connection_kwargs['port']}')
        return f'{self.__class__.__name__}<sentinels=[{','.join(sentinel_addresses)}]>'

    def check_master_state(self, state: dict, service_name: str) -> bool:
        if not state['is_master'] or state['is_sdown'] or state['is_odown']:
            return False
        if state['num-other-sentinels'] < self.min_other_sentinels:
            return False
        return True

    async def discover_master(self, service_name: str):
        """
        Asks sentinel servers for the AsyncKeyDB master's address corresponding
        to the service labeled ``service_name``.

        Returns a pair (address, port) or raises MasterNotFoundError if no
        master is found.
        """
        for sentinel_no, sentinel in enumerate(self.sentinels):
            try:
                masters = await sentinel.sentinel_masters()
            except (ConnectionError, TimeoutError):
                continue
            state = masters.get(service_name)
            if state and self.check_master_state(state, service_name):
                self.sentinels[0], self.sentinels[sentinel_no] = (sentinel, self.sentinels[0])
                return (state['ip'], state['port'])
        raise MasterNotFoundError(f'No master found for {service_name!r}')

    def filter_slaves(self, slaves: Iterable[Mapping]) -> Sequence[Tuple[EncodableT, EncodableT]]:
        """Remove slaves that are in an ODOWN or SDOWN state"""
        slaves_alive = []
        for slave in slaves:
            if slave['is_odown'] or slave['is_sdown']:
                continue
            slaves_alive.append((slave['ip'], slave['port']))
        return slaves_alive

    async def discover_slaves(self, service_name: str) -> Sequence[Tuple[EncodableT, EncodableT]]:
        """Returns a list of alive slaves for service ``service_name``"""
        for sentinel in self.sentinels:
            try:
                slaves = await sentinel.sentinel_slaves(service_name)
            except (ConnectionError, ResponseError, TimeoutError):
                continue
            slaves = self.filter_slaves(slaves)
            if slaves:
                return slaves
        return []

    def master_for(self, service_name: str, keydb_class: Type[AsyncKeyDB]=AsyncKeyDB, connection_pool_class: Type[AsyncSentinelConnectionPool]=AsyncSentinelConnectionPool, **kwargs):
        """
        Returns a redis client instance for the ``service_name`` master.

        A :py:class:`~redis.sentinel.SentinelConnectionPool` class is
        used to retrieve the master's address before establishing a new
        connection.

        NOTE: If the master's address has changed, any cached connections to
        the old master are closed.

        By default clients will be a :py:class:`~redis.AsyncKeyDB` instance.
        Specify a different class to the ``keydb_class`` argument if you
        desire something different.

        The ``connection_pool_class`` specifies the connection pool to
        use.  The :py:class:`~redis.sentinel.SentinelConnectionPool`
        will be used by default.

        All other keyword arguments are merged with any connection_kwargs
        passed to this class and passed to the connection pool as keyword
        arguments to be used to initialize AsyncKeyDB connections.
        """
        kwargs['is_master'] = True
        connection_kwargs = dict(self.connection_kwargs)
        connection_kwargs.update(kwargs)
        return keydb_class(connection_pool=connection_pool_class(service_name, self, **connection_kwargs))

    def slave_for(self, service_name: str, keydb_class: Type[AsyncKeyDB]=AsyncKeyDB, connection_pool_class: Type[AsyncSentinelConnectionPool]=AsyncSentinelConnectionPool, **kwargs):
        """
        Returns redis client instance for the ``service_name`` slave(s).

        A SentinelConnectionPool class is used to retrieve the slave's
        address before establishing a new connection.

        By default clients will be a :py:class:`~redis.AsyncKeyDB` instance.
        Specify a different class to the ``keydb_class`` argument if you
        desire something different.

        The ``connection_pool_class`` specifies the connection pool to use.
        The SentinelConnectionPool will be used by default.

        All other keyword arguments are merged with any connection_kwargs
        passed to this class and passed to the connection pool as keyword
        arguments to be used to initialize AsyncKeyDB connections.
        """
        kwargs['is_master'] = False
        connection_kwargs = dict(self.connection_kwargs)
        connection_kwargs.update(kwargs)
        return keydb_class(connection_pool=connection_pool_class(service_name, self, **connection_kwargs))