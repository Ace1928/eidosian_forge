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
def _response_closed(self, stream_id: int) -> None:
    self._max_streams_semaphore.release()
    del self._events[stream_id]
    with self._state_lock:
        if self._connection_terminated and (not self._events):
            self.close()
        elif self._state == HTTPConnectionState.ACTIVE and (not self._events):
            self._state = HTTPConnectionState.IDLE
            if self._keepalive_expiry is not None:
                now = time.monotonic()
                self._expire_at = now + self._keepalive_expiry
            if self._used_all_stream_ids:
                self.close()