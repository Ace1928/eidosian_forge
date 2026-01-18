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
def _read_incoming_data(self, request: Request) -> typing.List[h2.events.Event]:
    timeouts = request.extensions.get('timeout', {})
    timeout = timeouts.get('read', None)
    if self._read_exception is not None:
        raise self._read_exception
    try:
        data = self._network_stream.read(self.READ_NUM_BYTES, timeout)
        if data == b'':
            raise RemoteProtocolError('Server disconnected')
    except Exception as exc:
        self._read_exception = exc
        self._connection_error = True
        raise exc
    events: typing.List[h2.events.Event] = self._h2_state.receive_data(data)
    return events