import asyncio
import collections
import random
import socket
import warnings
from typing import (
from aiokeydb.v1.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from aiokeydb.v1.asyncio.core import ResponseCallbackT
from aiokeydb.v1.asyncio.connection import (
from aiokeydb.v1.asyncio.parser import CommandsParser
from aiokeydb.v1.core import EMPTY_RESPONSE, NEVER_DECODE, AbstractKeyDB
from aiokeydb.v1.cluster import (
from aiokeydb.v1.commands import READ_COMMANDS, AsyncKeyDBClusterCommands
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.typing import AnyKeyT, EncodableT, KeyT
from aiokeydb.v1.utils import dict_merge, safe_str, str_if_bytes
class AsyncClusterNode:
    """
    Create a new AsyncClusterNode.

    Each AsyncClusterNode manages multiple :class:`~redis.asyncio.connection.AsyncConnection`
    objects for the (host, port).
    """
    __slots__ = ('_connections', '_free', 'connection_class', 'connection_kwargs', 'host', 'max_connections', 'name', 'port', 'response_callbacks', 'server_type')

    def __init__(self, host: str, port: Union[str, int], server_type: Optional[str]=None, *, max_connections: int=2 ** 31, connection_class: Type[AsyncConnection]=AsyncConnection, **connection_kwargs: Any) -> None:
        if host == 'localhost':
            host = socket.gethostbyname(host)
        connection_kwargs['host'] = host
        connection_kwargs['port'] = port
        self.host = host
        self.port = port
        self.name = get_node_name(host, port)
        self.server_type = server_type
        self.max_connections = max_connections
        self.connection_class = connection_class
        self.connection_kwargs = connection_kwargs
        self.response_callbacks = connection_kwargs.pop('response_callbacks', {})
        self._connections: List[AsyncConnection] = []
        self._free: Deque[AsyncConnection] = collections.deque(maxlen=self.max_connections)

    def __repr__(self) -> str:
        return f'[host={self.host}, port={self.port}, name={self.name}, server_type={self.server_type}]'

    def __eq__(self, obj: Any) -> bool:
        return isinstance(obj, AsyncClusterNode) and obj.name == self.name
    _DEL_MESSAGE = 'Unclosed AsyncClusterNode object'

    def __del__(self) -> None:
        for connection in self._connections:
            if connection.is_connected:
                warnings.warn(f'{self._DEL_MESSAGE} {self!r}', ResourceWarning, source=self)
                try:
                    context = {'client': self, 'message': self._DEL_MESSAGE}
                    asyncio.get_running_loop().call_exception_handler(context)
                except RuntimeError:
                    ...
                break

    async def disconnect(self) -> None:
        ret = await asyncio.gather(*(asyncio.ensure_future(connection.disconnect()) for connection in self._connections), return_exceptions=True)
        exc = next((res for res in ret if isinstance(res, Exception)), None)
        if exc:
            raise exc

    def acquire_connection(self) -> AsyncConnection:
        try:
            return self._free.popleft()
        except IndexError:
            if len(self._connections) < self.max_connections:
                connection = self.connection_class(**self.connection_kwargs)
                self._connections.append(connection)
                return connection
            raise MaxConnectionsError()

    async def parse_response(self, connection: AsyncConnection, command: str, **kwargs: Any) -> Any:
        try:
            if NEVER_DECODE in kwargs:
                response = await connection.read_response_without_lock(disable_decoding=True)
            else:
                response = await connection.read_response_without_lock()
        except ResponseError:
            if EMPTY_RESPONSE in kwargs:
                return kwargs[EMPTY_RESPONSE]
            raise
        if command in self.response_callbacks:
            return self.response_callbacks[command](response, **kwargs)
        return response

    async def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        connection = self.acquire_connection()
        await connection.send_packed_command(connection.pack_command(*args), False)
        try:
            return await self.parse_response(connection, args[0], **kwargs)
        finally:
            self._free.append(connection)

    async def execute_pipeline(self, commands: List['PipelineCommand']) -> bool:
        connection = self.acquire_connection()
        await connection.send_packed_command(connection.pack_commands((cmd.args for cmd in commands)), False)
        ret = False
        for cmd in commands:
            try:
                cmd.result = await self.parse_response(connection, cmd.args[0], **cmd.kwargs)
            except Exception as e:
                cmd.result = e
                ret = True
        self._free.append(connection)
        return ret