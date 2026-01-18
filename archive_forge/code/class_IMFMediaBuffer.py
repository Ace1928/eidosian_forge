import os
import platform
import warnings
from pyglet import image
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32 import com
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.media import Source
from pyglet.media.codecs import AudioFormat, AudioData, VideoFormat, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class IMFMediaBuffer(com.pIUnknown):
    _methods_ = [('Lock', com.STDMETHOD(POINTER(POINTER(BYTE)), POINTER(DWORD), POINTER(DWORD))), ('Unlock', com.STDMETHOD()), ('GetCurrentLength', com.STDMETHOD(POINTER(DWORD))), ('SetCurrentLength', com.STDMETHOD(DWORD)), ('GetMaxLength', com.STDMETHOD(POINTER(DWORD)))]