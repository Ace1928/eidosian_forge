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
def _update_and_get_timing_info(self) -> Optional[pa.pa_timing_info]:
    self.stream.update_timing_info().wait().delete()
    return self.stream.get_timing_info()