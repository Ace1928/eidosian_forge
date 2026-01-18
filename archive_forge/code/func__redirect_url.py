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
def _redirect_url(self, request: Request, response: Response) -> URL:
    """
        Return the URL for the redirect to follow.
        """
    location = response.headers['Location']
    try:
        url = URL(location)
    except InvalidURL as exc:
        raise RemoteProtocolError(f'Invalid URL in location header: {exc}.', request=request) from None
    if url.scheme and (not url.host):
        url = url.copy_with(host=request.url.host)
    if url.is_relative_url:
        url = request.url.join(url)
    if request.url.fragment and (not url.fragment):
        url = url.copy_with(fragment=request.url.fragment)
    return url