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
def _receive_remote_settings_change(self, event: h2.events.Event) -> None:
    max_concurrent_streams = event.changed_settings.get(h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS)
    if max_concurrent_streams:
        new_max_streams = min(max_concurrent_streams.new_value, self._h2_state.local_settings.max_concurrent_streams)
        if new_max_streams and new_max_streams != self._max_streams:
            while new_max_streams > self._max_streams:
                self._max_streams_semaphore.release()
                self._max_streams += 1
            while new_max_streams < self._max_streams:
                self._max_streams_semaphore.acquire()
                self._max_streams -= 1