import enum
import logging
import time
import types
import typing
import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings
from .._backends.base import NetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import Lock, Semaphore, ShieldCancellation
from .._trace import Trace
from .interfaces import ConnectionInterface
def _send_request_headers(self, request: Request, stream_id: int) -> None:
    """
        Send the request headers to a given stream ID.
        """
    end_stream = not has_body_headers(request)
    authority = [v for k, v in request.headers if k.lower() == b'host'][0]
    headers = [(b':method', request.method), (b':authority', authority), (b':scheme', request.url.scheme), (b':path', request.url.target)] + [(k.lower(), v) for k, v in request.headers if k.lower() not in (b'host', b'transfer-encoding')]
    self._h2_state.send_headers(stream_id, headers, end_stream=end_stream)
    self._h2_state.increment_flow_control_window(2 ** 24, stream_id=stream_id)
    self._write_outgoing_data(request)