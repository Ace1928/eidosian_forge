import asyncio
import logging
import warnings
from functools import partial, update_wrapper
from typing import (
from aiosignal import Signal
from frozenlist import FrozenList
from . import hdrs
from .abc import (
from .helpers import DEBUG, AppKey
from .http_parser import RawRequestMessage
from .log import web_logger
from .streams import StreamReader
from .typedefs import Middleware
from .web_exceptions import NotAppKeyWarning
from .web_log import AccessLogger
from .web_middlewares import _fix_request_current_app
from .web_protocol import RequestHandler
from .web_request import Request
from .web_response import StreamResponse
from .web_routedef import AbstractRouteDef
from .web_server import Server
from .web_urldispatcher import (
def _set_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
    if loop is None:
        loop = asyncio.get_event_loop()
    if self._loop is not None and self._loop is not loop:
        raise RuntimeError('web.Application instance initialized with different loop')
    self._loop = loop
    if self._debug is ...:
        self._debug = loop.get_debug()
    for subapp in self._subapps:
        subapp._set_loop(loop)