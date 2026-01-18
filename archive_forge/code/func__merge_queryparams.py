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
def _merge_queryparams(self, params: QueryParamTypes | None=None) -> QueryParamTypes | None:
    """
        Merge a queryparams argument together with any queryparams on the client,
        to create the queryparams used for the outgoing request.
        """
    if params or self.params:
        merged_queryparams = QueryParams(self.params)
        return merged_queryparams.merge(params)
    return params