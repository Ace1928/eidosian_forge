import enum
import logging
import ssl
import time
from types import TracebackType
from typing import (
import h11
from .._backends.base import AsyncNetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import AsyncLock, AsyncShieldCancellation
from .._trace import Trace
from .interfaces import AsyncConnectionInterface
class AsyncHTTP11UpgradeStream(AsyncNetworkStream):

    def __init__(self, stream: AsyncNetworkStream, leading_data: bytes) -> None:
        self._stream = stream
        self._leading_data = leading_data

    async def read(self, max_bytes: int, timeout: Optional[float]=None) -> bytes:
        if self._leading_data:
            buffer = self._leading_data[:max_bytes]
            self._leading_data = self._leading_data[max_bytes:]
            return buffer
        else:
            return await self._stream.read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: Optional[float]=None) -> None:
        await self._stream.write(buffer, timeout)

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(self, ssl_context: ssl.SSLContext, server_hostname: Optional[str]=None, timeout: Optional[float]=None) -> AsyncNetworkStream:
        return await self._stream.start_tls(ssl_context, server_hostname, timeout)

    def get_extra_info(self, info: str) -> Any:
        return self._stream.get_extra_info(info)