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
def _write_outgoing_data(self, request: Request) -> None:
    timeouts = request.extensions.get('timeout', {})
    timeout = timeouts.get('write', None)
    with self._write_lock:
        data_to_send = self._h2_state.data_to_send()
        if self._write_exception is not None:
            raise self._write_exception
        try:
            self._network_stream.write(data_to_send, timeout)
        except Exception as exc:
            self._write_exception = exc
            self._connection_error = True
            raise exc