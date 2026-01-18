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
def _send_handling_redirects(self, request: Request, follow_redirects: bool, history: list[Response]) -> Response:
    while True:
        if len(history) > self.max_redirects:
            raise TooManyRedirects('Exceeded maximum allowed redirects.', request=request)
        for hook in self._event_hooks['request']:
            hook(request)
        response = self._send_single_request(request)
        try:
            for hook in self._event_hooks['response']:
                hook(response)
            response.history = list(history)
            if not response.has_redirect_location:
                return response
            request = self._build_redirect_request(request, response)
            history = history + [response]
            if follow_redirects:
                response.read()
            else:
                response.next_request = request
                return response
        except BaseException as exc:
            response.close()
            raise exc