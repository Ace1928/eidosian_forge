from collections import deque
import ctypes
import threading
from typing import Deque, Optional, TYPE_CHECKING
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
from . import lib_pulseaudio as pa
from .interface import PulseAudioMainloop
def _maybe_write_pending(self) -> None:
    with self._audio_data_lock:
        if self._pending_bytes == 0 or self._audio_data_buffer.available == 0:
            return
        written = self._write_to_stream(self._pending_bytes)
        self._pending_bytes -= written
        if not self._has_underrun:
            return
        self._has_underrun = False
    self.stream.trigger().wait().delete()