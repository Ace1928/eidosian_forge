import asyncio
from contextlib import suppress
from typing import Any, Optional, Tuple
from .base_protocol import BaseProtocol
from .client_exceptions import (
from .helpers import BaseTimerContext, status_code_must_be_empty_body
from .http import HttpResponseParser, RawResponseMessage
from .streams import EMPTY_PAYLOAD, DataQueue, StreamReader
def _reschedule_timeout(self) -> None:
    timeout = self._read_timeout
    if self._read_timeout_handle is not None:
        self._read_timeout_handle.cancel()
    if timeout:
        self._read_timeout_handle = self._loop.call_later(timeout, self._on_read_timeout)
    else:
        self._read_timeout_handle = None