import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
def _create_buffer_desc(wave_format, buffer_size):
    dsbdesc = lib.DSBUFFERDESC()
    dsbdesc.dwSize = ctypes.sizeof(dsbdesc)
    dsbdesc.dwFlags = lib.DSBCAPS_GLOBALFOCUS | lib.DSBCAPS_GETCURRENTPOSITION2 | lib.DSBCAPS_CTRLFREQUENCY | lib.DSBCAPS_CTRLVOLUME
    if wave_format.nChannels == 1:
        dsbdesc.dwFlags |= lib.DSBCAPS_CTRL3D
    dsbdesc.dwBufferBytes = buffer_size
    dsbdesc.lpwfxFormat = ctypes.pointer(wave_format)
    return dsbdesc