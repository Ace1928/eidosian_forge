import enum
import logging
import ssl
import time
from types import TracebackType
from typing import (
import h11
from .._backends.base import NetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import Lock, ShieldCancellation
from .._trace import Trace
from .interfaces import ConnectionInterface
def _receive_response_headers(self, request: Request) -> Tuple[bytes, int, bytes, List[Tuple[bytes, bytes]], bytes]:
    timeouts = request.extensions.get('timeout', {})
    timeout = timeouts.get('read', None)
    while True:
        event = self._receive_event(timeout=timeout)
        if isinstance(event, h11.Response):
            break
        if isinstance(event, h11.InformationalResponse) and event.status_code == 101:
            break
    http_version = b'HTTP/' + event.http_version
    headers = event.headers.raw_items()
    trailing_data, _ = self._h11_state.trailing_data
    return (http_version, event.status_code, event.reason, headers, trailing_data)