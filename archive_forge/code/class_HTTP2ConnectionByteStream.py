import enum
import logging
import time
import types
import typing
import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings
from .._backends.base import AsyncNetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import AsyncLock, AsyncSemaphore, AsyncShieldCancellation
from .._trace import Trace
from .interfaces import AsyncConnectionInterface
class HTTP2ConnectionByteStream:

    def __init__(self, connection: AsyncHTTP2Connection, request: Request, stream_id: int) -> None:
        self._connection = connection
        self._request = request
        self._stream_id = stream_id
        self._closed = False

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        kwargs = {'request': self._request, 'stream_id': self._stream_id}
        try:
            async with Trace('receive_response_body', logger, self._request, kwargs):
                async for chunk in self._connection._receive_response_body(request=self._request, stream_id=self._stream_id):
                    yield chunk
        except BaseException as exc:
            with AsyncShieldCancellation():
                await self.aclose()
            raise exc

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            kwargs = {'stream_id': self._stream_id}
            async with Trace('response_closed', logger, self._request, kwargs):
                await self._connection._response_closed(stream_id=self._stream_id)