from __future__ import annotations
import datetime
import enum
import logging
import typing
import warnings
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType
from .__version__ import __version__
from ._auth import Auth, BasicAuth, FunctionAuth
from ._config import (
from ._decoders import SUPPORTED_DECODERS
from ._exceptions import (
from ._models import Cookies, Headers, Request, Response
from ._status_codes import codes
from ._transports.asgi import ASGITransport
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._transports.default import AsyncHTTPTransport, HTTPTransport
from ._transports.wsgi import WSGITransport
from ._types import (
from ._urls import URL, QueryParams
from ._utils import (
class BoundAsyncStream(AsyncByteStream):
    """
    An async byte stream that is bound to a given response instance, and that
    ensures the `response.elapsed` is set once the response is closed.
    """

    def __init__(self, stream: AsyncByteStream, response: Response, timer: Timer) -> None:
        self._stream = stream
        self._response = response
        self._timer = timer

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        seconds = await self._timer.async_elapsed()
        self._response.elapsed = datetime.timedelta(seconds=seconds)
        await self._stream.aclose()