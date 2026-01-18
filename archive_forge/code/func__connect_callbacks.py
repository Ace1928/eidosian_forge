import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def _connect_callbacks(self) -> None:
    s = self._pa_stream
    pa.pa_stream_set_underflow_callback(s, self._cb_underflow, None)
    pa.pa_stream_set_write_callback(s, self._cb_write, None)
    pa.pa_stream_set_state_callback(s, self._cb_state, None)
    pa.pa_stream_set_moved_callback(s, self._cb_moved, None)