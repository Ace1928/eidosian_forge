import math
import ctypes
from . import interface
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
def _get_used_buffer_space(self):
    return max(self._write_cursor - self._play_cursor, 0)