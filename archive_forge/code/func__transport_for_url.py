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
def _transport_for_url(self, url: URL) -> AsyncBaseTransport:
    """
        Returns the transport instance that should be used for a given URL.
        This will either be the standard connection pool, or a proxy.
        """
    for pattern, transport in self._mounts.items():
        if pattern.matches(url):
            return self._transport if transport is None else transport
    return self._transport