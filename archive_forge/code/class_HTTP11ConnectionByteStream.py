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
class HTTP11ConnectionByteStream:

    def __init__(self, connection: AsyncHTTP11Connection, request: Request) -> None:
        self._connection = connection
        self._request = request
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        kwargs = {'request': self._request}
        try:
            async with Trace('receive_response_body', logger, self._request, kwargs):
                async for chunk in self._connection._receive_response_body(**kwargs):
                    yield chunk
        except BaseException as exc:
            with AsyncShieldCancellation():
                await self.aclose()
            raise exc

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            async with Trace('response_closed', logger, self._request):
                await self._connection._response_closed()