import asyncio
import asyncio.streams
import traceback
import warnings
from collections import deque
from contextlib import suppress
from html import escape as html_escape
from http import HTTPStatus
from logging import Logger
from typing import (
import attr
import yarl
from .abc import AbstractAccessLogger, AbstractStreamWriter
from .base_protocol import BaseProtocol
from .helpers import ceil_timeout
from .http import (
from .log import access_logger, server_logger
from .streams import EMPTY_PAYLOAD, StreamReader
from .tcp_helpers import tcp_keepalive
from .web_exceptions import HTTPException
from .web_log import AccessLogger
from .web_request import BaseRequest
from .web_response import Response, StreamResponse
def _process_keepalive(self) -> None:
    if self._force_close or not self._keepalive:
        return
    next = self._keepalive_time + self._keepalive_timeout
    if self._waiter:
        if self._loop.time() > next:
            self.force_close()
            return
    self._keepalive_handle = self._loop.call_later(self.KEEPALIVE_RESCHEDULE_DELAY, self._process_keepalive)