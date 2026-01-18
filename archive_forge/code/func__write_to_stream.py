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
def _write_to_stream(self, nbytes: int) -> int:
    data_ptr, bytes_accepted = self.stream.begin_write(nbytes)
    bytes_written = self._audio_data_buffer.memmove(data_ptr.value, bytes_accepted)
    if bytes_written == 0:
        self.stream.cancel_write()
    else:
        self.stream.write(data_ptr, bytes_written, pa.PA_SEEK_RELATIVE)
    assert _debug(f'PulseAudioPlayer: Wrote {bytes_written}/{nbytes}')
    return bytes_written