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
def _send_single_request(self, request: Request) -> Response:
    """
        Sends a single request, without handling any redirections.
        """
    transport = self._transport_for_url(request.url)
    timer = Timer()
    timer.sync_start()
    if not isinstance(request.stream, SyncByteStream):
        raise RuntimeError('Attempted to send an async request with a sync Client instance.')
    with request_context(request=request):
        response = transport.handle_request(request)
    assert isinstance(response.stream, SyncByteStream)
    response.request = request
    response.stream = BoundSyncStream(response.stream, response=response, timer=timer)
    self.cookies.extract_cookies(response)
    response.default_encoding = self._default_encoding
    logger.info('HTTP Request: %s %s "%s %d %s"', request.method, request.url, response.http_version, response.status_code, response.reason_phrase)
    return response