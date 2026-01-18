import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2_VOICE_SENDS(ctypes.Structure):
    _fields_ = [('SendCount', UINT32), ('pSends', c_void_p)]