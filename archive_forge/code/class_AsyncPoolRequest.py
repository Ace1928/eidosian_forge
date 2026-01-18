import ssl
import sys
from types import TracebackType
from typing import AsyncIterable, AsyncIterator, Iterable, List, Optional, Type
from .._backends.auto import AutoBackend
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend
from .._exceptions import ConnectionNotAvailable, UnsupportedProtocol
from .._models import Origin, Request, Response
from .._synchronization import AsyncEvent, AsyncShieldCancellation, AsyncThreadLock
from .connection import AsyncHTTPConnection
from .interfaces import AsyncConnectionInterface, AsyncRequestInterface
class AsyncPoolRequest:

    def __init__(self, request: Request) -> None:
        self.request = request
        self.connection: Optional[AsyncConnectionInterface] = None
        self._connection_acquired = AsyncEvent()

    def assign_to_connection(self, connection: Optional[AsyncConnectionInterface]) -> None:
        self.connection = connection
        self._connection_acquired.set()

    def clear_connection(self) -> None:
        self.connection = None
        self._connection_acquired = AsyncEvent()

    async def wait_for_connection(self, timeout: Optional[float]=None) -> AsyncConnectionInterface:
        if self.connection is None:
            await self._connection_acquired.wait(timeout=timeout)
        assert self.connection is not None
        return self.connection

    def is_queued(self) -> bool:
        return self.connection is None