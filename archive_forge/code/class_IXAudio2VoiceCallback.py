import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class IXAudio2VoiceCallback(com.Interface):
    _methods_ = [('OnVoiceProcessingPassStart', com.VOIDMETHOD(UINT32)), ('OnVoiceProcessingPassEnd', com.VOIDMETHOD()), ('OnStreamEnd', com.VOIDMETHOD()), ('OnBufferStart', com.VOIDMETHOD(ctypes.c_void_p)), ('OnBufferEnd', com.VOIDMETHOD(ctypes.c_void_p)), ('OnLoopEnd', com.VOIDMETHOD(ctypes.c_void_p)), ('OnVoiceError', com.VOIDMETHOD(ctypes.c_void_p, HRESULT))]