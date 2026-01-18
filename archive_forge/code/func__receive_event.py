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
def _receive_event(self, timeout: Optional[float]=None) -> Union[h11.Event, Type[h11.PAUSED]]:
    while True:
        with map_exceptions({h11.RemoteProtocolError: RemoteProtocolError}):
            event = self._h11_state.next_event()
        if event is h11.NEED_DATA:
            data = self._network_stream.read(self.READ_NUM_BYTES, timeout=timeout)
            if data == b'' and self._h11_state.their_state == h11.SEND_RESPONSE:
                msg = 'Server disconnected without sending a response.'
                raise RemoteProtocolError(msg)
            self._h11_state.receive_data(data)
        else:
            return event