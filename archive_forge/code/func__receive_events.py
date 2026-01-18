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
def _receive_events(self, request: Request, stream_id: typing.Optional[int]=None) -> None:
    """
        Read some data from the network until we see one or more events
        for a given stream ID.
        """
    with self._read_lock:
        if self._connection_terminated is not None:
            last_stream_id = self._connection_terminated.last_stream_id
            if stream_id and last_stream_id and (stream_id > last_stream_id):
                self._request_count -= 1
                raise ConnectionNotAvailable()
            raise RemoteProtocolError(self._connection_terminated)
        if stream_id is None or not self._events.get(stream_id):
            events = self._read_incoming_data(request)
            for event in events:
                if isinstance(event, h2.events.RemoteSettingsChanged):
                    with Trace('receive_remote_settings', logger, request) as trace:
                        self._receive_remote_settings_change(event)
                        trace.return_value = event
                elif isinstance(event, (h2.events.ResponseReceived, h2.events.DataReceived, h2.events.StreamEnded, h2.events.StreamReset)):
                    if event.stream_id in self._events:
                        self._events[event.stream_id].append(event)
                elif isinstance(event, h2.events.ConnectionTerminated):
                    self._connection_terminated = event
    self._write_outgoing_data(request)