import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
class _MainloopLock:

    def __init__(self, mainloop: 'PulseAudioMainloop') -> None:
        self.mainloop = mainloop

    def __enter__(self):
        self.mainloop.lock_()

    def __exit__(self, _exc_type, _ecx_value, _tb):
        self.mainloop.unlock()