import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
def _create_primary_buffer_desc():
    """Primary buffer with 3D and volume capabilities"""
    buffer_desc = lib.DSBUFFERDESC()
    buffer_desc.dwSize = ctypes.sizeof(buffer_desc)
    buffer_desc.dwFlags = lib.DSBCAPS_CTRL3D | lib.DSBCAPS_CTRLVOLUME | lib.DSBCAPS_PRIMARYBUFFER
    return buffer_desc