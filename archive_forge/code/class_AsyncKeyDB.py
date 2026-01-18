import logging
import asyncio
import copy
import inspect
import re
import warnings
from typing import (
from aiokeydb.v1.asyncio.connection import (
from aiokeydb.v1.asyncio.lock import AsyncLock
from aiokeydb.v1.asyncio.retry import Retry
from aiokeydb.v1.core import (
from aiokeydb.v1.commands import (
from aiokeydb.v1.compat import Protocol, TypedDict
from aiokeydb.v1.credentials import CredentialProvider
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.typing import ChannelT, EncodableT, KeyT
from aiokeydb.v1.utils import safe_str, str_if_bytes
class AsyncKeyDB(AbstractKeyDB, RedisModuleCommands, AsyncCoreCommands, AsyncSentinelCommands):
    """
    Implementation of the KeyDB protocol.

    This abstract class provides a Python interface to all KeyDB commands
    and an implementation of the KeyDB protocol.

    Pipelines derive from this, implementing how
    the commands are sent and received to the KeyDB server. Based on
    configuration, an instance will either use a AsyncConnectionPool, or
    AsyncConnection object to talk to redis.
    """
    response_callbacks: MutableMapping[Union[str, bytes], ResponseCallbackT]

    @property
    def is_async(self):
        return True

    @classmethod
    def from_url(cls, url: str, single_connection_client: bool=False, connection_pool_cls: Optional[Type[AsyncConnectionPool]]=None, **kwargs):
        """
        Return a KeyDB client object configured from the given URL

        For example::
            keydb://[[username]:[password]]@localhost:6379/0
            keydbs://[[username]:[password]]@localhost:6379/0
            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[[username]:[password]]@/path/to/socket.sock?db=0

        Five URL schemes are supported:
        - `keydb://` creates a TCP socket connection.
        - `keydbs://` creates a SSL wrapped TCP socket connection.
        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>
        - ``unix://``: creates a Unix Domain Socket connection.

        The username, password, hostname, path and all querystring values
        are passed through urllib.parse.unquote in order to replace any
        percent-encoded values with their corresponding characters.

        There are several ways to specify a database number. The first value
        found will be used:
            1. A ``db`` querystring option, e.g. redis://localhost?db=0
            2. If using the redis:// or rediss:// schemes, the path argument
               of the url, e.g. redis://localhost/0
            3. A ``db`` keyword argument to this function.

        If none of these options are specified, the default db=0 is used.

        All querystring options are cast to their appropriate Python types.
        Boolean arguments can be specified with string values "True"/"False"
        or "Yes"/"No". Values that cannot be properly cast cause a
        ``ValueError`` to be raised. Once parsed, the querystring arguments
        and keyword arguments are passed to the ``AsyncConnectionPool``'s
        class initializer. In the case of conflicting arguments, querystring
        arguments always win.

        """
        if connection_pool_cls is None:
            connection_pool_cls = AsyncConnectionPool
        connection_pool = connection_pool_cls.from_url(url, **kwargs)
        return cls(connection_pool=connection_pool, single_connection_client=single_connection_client)

    def __init__(self, *, host: str='localhost', port: int=6379, db: Union[str, int]=0, password: Optional[str]=None, socket_timeout: Optional[float]=None, socket_connect_timeout: Optional[float]=None, socket_keepalive: Optional[bool]=None, socket_keepalive_options: Optional[Mapping[int, Union[int, bytes]]]=None, connection_pool: Optional[AsyncConnectionPool]=None, unix_socket_path: Optional[str]=None, encoding: str='utf-8', encoding_errors: str='strict', decode_responses: bool=False, retry_on_timeout: bool=False, retry_on_error: Optional[list]=None, ssl: bool=False, ssl_keyfile: Optional[str]=None, ssl_certfile: Optional[str]=None, ssl_cert_reqs: str='required', ssl_ca_certs: Optional[str]=None, ssl_ca_data: Optional[str]=None, ssl_check_hostname: bool=False, max_connections: Optional[int]=None, single_connection_client: bool=False, health_check_interval: int=0, client_name: Optional[str]=None, username: Optional[str]=None, retry: Optional[Retry]=None, auto_close_connection_pool: bool=True, keydb_connect_func=None, credential_provider: Optional[CredentialProvider]=None):
        """
        Initialize a new KeyDB client.
        To specify a retry policy for specific errors, first set
        `retry_on_error` to a list of the error/s to retry on, then set
        `retry` to a valid `Retry` object.
        To retry on TimeoutError, `retry_on_timeout` can also be set to `True`.
        """
        kwargs: Dict[str, Any]
        self.auto_close_connection_pool = auto_close_connection_pool if connection_pool is None else False
        if not connection_pool:
            if not retry_on_error:
                retry_on_error = []
            if retry_on_timeout is True:
                retry_on_error.append(TimeoutError)
            kwargs = {'db': db, 'username': username, 'password': password, 'credential_provider': credential_provider, 'socket_timeout': socket_timeout, 'encoding': encoding, 'encoding_errors': encoding_errors, 'decode_responses': decode_responses, 'retry_on_timeout': retry_on_timeout, 'retry_on_error': retry_on_error, 'retry': copy.deepcopy(retry), 'max_connections': max_connections, 'health_check_interval': health_check_interval, 'client_name': client_name, 'keydb_connect_func': keydb_connect_func}
            if unix_socket_path is not None:
                kwargs.update({'path': unix_socket_path, 'connection_class': AsyncUnixDomainSocketConnection})
            else:
                kwargs.update({'host': host, 'port': port, 'socket_connect_timeout': socket_connect_timeout, 'socket_keepalive': socket_keepalive, 'socket_keepalive_options': socket_keepalive_options})
                if ssl:
                    kwargs.update({'connection_class': AsyncSSLConnection, 'ssl_keyfile': ssl_keyfile, 'ssl_certfile': ssl_certfile, 'ssl_cert_reqs': ssl_cert_reqs, 'ssl_ca_certs': ssl_ca_certs, 'ssl_ca_data': ssl_ca_data, 'ssl_check_hostname': ssl_check_hostname})
            connection_pool = AsyncConnectionPool(**kwargs)
        self.connection_pool = connection_pool
        self.single_connection_client = single_connection_client
        self.connection: Optional[AsyncConnection] = None
        self.response_callbacks = CaseInsensitiveDict(self.__class__.RESPONSE_CALLBACKS)
        self._single_conn_lock = asyncio.Lock()

    def __repr__(self):
        return f'{self.__class__.__name__}<{self.connection_pool!r}>'

    def __await__(self):
        return self.initialize().__await__()

    async def initialize(self: _KeyDBT) -> _KeyDBT:
        if self.single_connection_client:
            async with self._single_conn_lock:
                if self.connection is None:
                    self.connection = await self.connection_pool.get_connection('_')
        return self

    def set_response_callback(self, command: str, callback: ResponseCallbackT):
        """Set a custom Response Callback"""
        self.response_callbacks[command] = callback

    def get_encoder(self):
        """Get the connection pool's encoder"""
        return self.connection_pool.get_encoder()

    def get_connection_kwargs(self):
        """Get the connection's key-word arguments"""
        return self.connection_pool.connection_kwargs

    def set_retry(self, retry: 'Retry') -> None:
        self.get_connection_kwargs().update({'retry': retry})
        self.connection_pool.set_retry(retry)

    def load_external_module(self, funcname, func):
        """
        This function can be used to add externally defined redis modules,
        and their namespaces to the redis client.

        funcname - A string containing the name of the function to create
        func - The function, being added to this class.

        ex: Assume that one has a custom redis module named foomod that
        creates command named 'foo.dothing' and 'foo.anotherthing' in redis.
        To load function functions into this namespace:

        from redis import KeyDB
        from foomodule import F
        r = KeyDB()
        r.load_external_module("foo", F)
        r.foo().dothing('your', 'arguments')

        For a concrete example see the reimport of the redisjson module in
        tests/test_connection.py::test_loading_external_modules
        """
        setattr(self, funcname, func)

    def pipeline(self, transaction: bool=True, shard_hint: Optional[str]=None) -> 'AsyncPipeline':
        """
        Return a new pipeline object that can queue multiple commands for
        later execution. ``transaction`` indicates whether all commands
        should be executed atomically. Apart from making a group of operations
        atomic, pipelines are useful for reducing the back-and-forth overhead
        between the client and server.
        """
        return AsyncPipeline(self.connection_pool, self.response_callbacks, transaction, shard_hint)

    async def transaction(self, func: Callable[['AsyncPipeline'], Union[Any, Awaitable[Any]]], *watches: KeyT, shard_hint: Optional[str]=None, value_from_callable: bool=False, watch_delay: Optional[float]=None):
        """
        Convenience method for executing the callable `func` as a transaction
        while watching all keys specified in `watches`. The 'func' callable
        should expect a single argument which is an AsyncPipeline object.
        """
        pipe: AsyncPipeline
        async with self.pipeline(True, shard_hint) as pipe:
            while True:
                try:
                    if watches:
                        await pipe.watch(*watches)
                    func_value = func(pipe)
                    if inspect.isawaitable(func_value):
                        func_value = await func_value
                    exec_value = await pipe.execute()
                    return func_value if value_from_callable else exec_value
                except WatchError:
                    if watch_delay is not None and watch_delay > 0:
                        await asyncio.sleep(watch_delay)
                    continue

    def lock(self, name: KeyT, timeout: Optional[float]=None, sleep: float=0.1, blocking_timeout: Optional[float]=None, lock_class: Optional[Type[AsyncLock]]=None, thread_local: bool=True) -> AsyncLock:
        """
        Return a new Lock object using key ``name`` that mimics
        the behavior of threading.Lock.

        If specified, ``timeout`` indicates a maximum life for the lock.
        By default, it will remain locked until release() is called.

        ``sleep`` indicates the amount of time to sleep per loop iteration
        when the lock is in blocking mode and another client is currently
        holding the lock.

        ``blocking_timeout`` indicates the maximum amount of time in seconds to
        spend trying to acquire the lock. A value of ``None`` indicates
        continue trying forever. ``blocking_timeout`` can be specified as a
        float or integer, both representing the number of seconds to wait.

        ``lock_class`` forces the specified lock implementation. Note that as
        of redis-py 3.0, the only lock class we implement is ``Lock`` (which is
        a Lua-based lock). So, it's unlikely you'll need this parameter, unless
        you have created your own custom lock class.

        ``thread_local`` indicates whether the lock token is placed in
        thread-local storage. By default, the token is placed in thread local
        storage so that a thread only sees its token, not a token set by
        another thread. Consider the following timeline:

            time: 0, thread-1 acquires `my-lock`, with a timeout of 5 seconds.
                     thread-1 sets the token to "abc"
            time: 1, thread-2 blocks trying to acquire `my-lock` using the
                     Lock instance.
            time: 5, thread-1 has not yet completed. redis expires the lock
                     key.
            time: 5, thread-2 acquired `my-lock` now that it's available.
                     thread-2 sets the token to "xyz"
            time: 6, thread-1 finishes its work and calls release(). if the
                     token is *not* stored in thread local storage, then
                     thread-1 would see the token value as "xyz" and would be
                     able to successfully release the thread-2's lock.

        In some use cases it's necessary to disable thread local storage. For
        example, if you have code where one thread acquires a lock and passes
        that lock instance to a worker thread to release later. If thread
        local storage isn't disabled in this case, the worker thread won't see
        the token set by the thread that acquired the lock. Our assumption
        is that these cases aren't common and as such default to using
        thread local storage."""
        if lock_class is None:
            lock_class = AsyncLock
        return lock_class(self, name, timeout=timeout, sleep=sleep, blocking_timeout=blocking_timeout, thread_local=thread_local)

    def pubsub(self, **kwargs) -> 'AsyncPubSub':
        """
        Return a Publish/Subscribe object. With this object, you can
        subscribe to channels and listen for messages that get published to
        them.
        """
        return AsyncPubSub(self.connection_pool, **kwargs)

    def monitor(self) -> 'AsyncMonitor':
        return AsyncMonitor(self.connection_pool)

    def client(self) -> 'AsyncKeyDB':
        return self.__class__(connection_pool=self.connection_pool, single_connection_client=True)

    async def __aenter__(self: _KeyDBT) -> _KeyDBT:
        return await self.initialize()

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()
    _DEL_MESSAGE = 'Unclosed KeyDB client'

    def __del__(self, _warnings: Any=warnings) -> None:
        if self.connection is not None:
            _warnings.warn(f'Unclosed client session {self!r}', ResourceWarning, source=self)
            context = {'client': self, 'message': self._DEL_MESSAGE}
            asyncio.get_event_loop().call_exception_handler(context)

    async def close(self, close_connection_pool: Optional[bool]=None) -> None:
        """
        Closes KeyDB client connection

        :param close_connection_pool: decides whether to close the connection pool used
        by this KeyDB client, overriding KeyDB.auto_close_connection_pool. By default,
        let KeyDB.auto_close_connection_pool decide whether to close the connection
        pool.
        """
        conn = self.connection
        if conn:
            self.connection = None
            await self.connection_pool.release(conn)
        if close_connection_pool or (close_connection_pool is None and self.auto_close_connection_pool):
            await self.connection_pool.disconnect()

    async def _send_command_parse_response(self, conn, command_name, *args, **options):
        """
        Send a command and parse the response
        """
        await conn.send_command(*args)
        return await self.parse_response(conn, command_name, **options)

    async def _disconnect_raise(self, conn: AsyncConnection, error: Exception):
        """
        Close the connection and raise an exception
        if retry_on_timeout is not set or the error
        is not a TimeoutError
        """
        await conn.disconnect()
        if conn.retry_on_error is None or isinstance(error, tuple(conn.retry_on_error)) is False:
            raise error

    async def execute_command(self, *args, **options):
        """Execute a command and return a parsed response"""
        await self.initialize()
        pool = self.connection_pool
        command_name = args[0]
        conn = self.connection or await pool.get_connection(command_name, **options)
        if self.single_connection_client:
            await self._single_conn_lock.acquire()
        try:
            return await conn.retry.call_with_retry(lambda: self._send_command_parse_response(conn, command_name, *args, **options), lambda error: self._disconnect_raise(conn, error))
        finally:
            if self.single_connection_client:
                self._single_conn_lock.release()
            if not self.connection:
                await pool.release(conn)

    async def parse_response(self, connection: 'AsyncConnection', command_name: Union[str, bytes], **options):
        """Parses a response from the Redis server"""
        try:
            if NEVER_DECODE in options:
                response = await connection.read_response(disable_decoding=True)
                options.pop(NEVER_DECODE)
            else:
                response = await connection.read_response()
        except ResponseError:
            if EMPTY_RESPONSE in options:
                return options[EMPTY_RESPONSE]
            raise
        if EMPTY_RESPONSE in options:
            options.pop(EMPTY_RESPONSE)
        if command_name in self.response_callbacks:
            command_name = cast(str, command_name)
            retval = self.response_callbacks[command_name](response, **options)
            return await retval if inspect.isawaitable(retval) else retval
        return response