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
class PoolByteStream:

    def __init__(self, stream: AsyncIterable[bytes], pool_request: AsyncPoolRequest, pool: AsyncConnectionPool) -> None:
        self._stream = stream
        self._pool_request = pool_request
        self._pool = pool
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for part in self._stream:
                yield part
        except BaseException as exc:
            await self.aclose()
            raise exc from None

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            with AsyncShieldCancellation():
                if hasattr(self._stream, 'aclose'):
                    await self._stream.aclose()
            with self._pool._optional_thread_lock:
                self._pool._requests.remove(self._pool_request)
                closing = self._pool._assign_requests_to_connections()
            await self._pool._close_connections(closing)