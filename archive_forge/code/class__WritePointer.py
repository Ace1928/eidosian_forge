import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
class _WritePointer:

    def __init__(self):
        self.audio_ptr_1 = ctypes.c_void_p()
        self.audio_length_1 = lib.DWORD()
        self.audio_ptr_2 = ctypes.c_void_p()
        self.audio_length_2 = lib.DWORD()