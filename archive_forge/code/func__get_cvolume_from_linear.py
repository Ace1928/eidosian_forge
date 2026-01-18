import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def _get_cvolume_from_linear(self, stream: 'PulseAudioStream', volume: float) -> pa.pa_cvolume:
    cvolume = pa.pa_cvolume()
    volume = pa.pa_sw_volume_from_linear(volume)
    pa.pa_cvolume_set(cvolume, stream.get_sample_spec().channels, volume)
    return cvolume