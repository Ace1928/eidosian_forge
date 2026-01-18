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
def _send_request_body(self, request: Request, stream_id: int) -> None:
    """
        Iterate over the request body sending it to a given stream ID.
        """
    if not has_body_headers(request):
        return
    assert isinstance(request.stream, typing.Iterable)
    for data in request.stream:
        self._send_stream_data(request, stream_id, data)
    self._send_end_stream(request, stream_id)