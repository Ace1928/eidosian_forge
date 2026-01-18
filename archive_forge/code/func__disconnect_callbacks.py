import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def _disconnect_callbacks(self) -> None:
    s = self._pa_stream
    pa.pa_stream_set_underflow_callback(s, pa.pa_stream_notify_cb_t(0), None)
    pa.pa_stream_set_write_callback(s, pa.pa_stream_request_cb_t(0), None)
    pa.pa_stream_set_state_callback(s, pa.pa_stream_notify_cb_t(0), None)
    pa.pa_stream_set_moved_callback(s, pa.pa_stream_notify_cb_t(0), None)