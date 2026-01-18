import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class IXAudio2EngineCallback(com.Interface):
    _methods_ = [('OnProcessingPassStart', com.VOIDMETHOD()), ('OnProcessingPassEnd', com.VOIDMETHOD()), ('OnCriticalError', com.VOIDMETHOD(HRESULT))]