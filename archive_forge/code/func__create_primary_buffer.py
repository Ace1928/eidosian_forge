import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
def _create_primary_buffer(self):
    return DirectSoundBuffer(self._create_native_buffer(_create_primary_buffer_desc()), None, 0)