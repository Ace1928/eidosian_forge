import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2_VOICE_DETAILS(Structure):
    _fields_ = [('CreationFlags', UINT32), ('ActiveFlags', UINT32), ('InputChannels', UINT32), ('InputSampleRate', UINT32)]