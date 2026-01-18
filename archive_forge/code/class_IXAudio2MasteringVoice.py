import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class IXAudio2MasteringVoice(IXAudio2Voice):
    _methods_ = [('GetChannelMask', com.STDMETHOD(POINTER(DWORD)))]