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
def _build_request_auth(self, request: Request, auth: AuthTypes | UseClientDefault | None=USE_CLIENT_DEFAULT) -> Auth:
    auth = self._auth if isinstance(auth, UseClientDefault) else self._build_auth(auth)
    if auth is not None:
        return auth
    username, password = (request.url.username, request.url.password)
    if username or password:
        return BasicAuth(username=username, password=password)
    return Auth()