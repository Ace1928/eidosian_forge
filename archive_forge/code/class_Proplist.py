import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
class Proplist:

    def __init__(self, ini_data: Optional[Dict[str, Union[bytes, str]]]=None) -> None:
        self._pl = pa.pa_proplist_new()
        if not self._pl:
            raise PulseAudioException(0, 'Failed creating proplist.')
        if ini_data is not None:
            for k, v in ini_data:
                self[k] = v

    def __setitem__(self, k, v):
        if isinstance(v, bytes):
            r = pa.pa_proplist_set(self._pl, k.encode('utf-8'), v, len(v))
        else:
            r = pa.pa_proplist_sets(self._pl, k.encode('utf-8'), v.encode('utf-8'))
        if r != 0:
            raise PulseAudioException(0, 'Error setting proplist entry.')

    def __delitem__(self, k):
        if pa.pa_proplist_unset(k) != 0:
            raise PulseAudioException(0, 'Error unsetting proplist entry.')

    def delete(self) -> None:
        pa.pa_proplist_free(self._pl)
        self._pl = None