import math
import ctypes
from . import interface
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
def _maybe_fill(self):
    if (used := self._get_used_buffer_space()) < self._buffered_data_comfortable_limit:
        self._refill(self.source.audio_format.align(self._buffer_size - used))