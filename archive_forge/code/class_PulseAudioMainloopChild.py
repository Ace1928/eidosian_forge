import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
class PulseAudioMainloopChild:

    def __init__(self, mainloop: PulseAudioMainloop):
        assert mainloop is not None
        self.mainloop = mainloop