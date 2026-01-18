import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def _state_callback(self, _stream, _userdata) -> None:
    self._refresh_state()
    assert _debug(f'PulseAudioStream: state changed to {self._state_name[self.state]}')
    self.mainloop.signal()