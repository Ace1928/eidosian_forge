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
class IMFSourceReader(com.pIUnknown):
    _methods_ = [('GetStreamSelection', com.STDMETHOD(DWORD, POINTER(BOOL))), ('SetStreamSelection', com.STDMETHOD(DWORD, BOOL)), ('GetNativeMediaType', com.STDMETHOD(DWORD, DWORD, POINTER(IMFMediaType))), ('GetCurrentMediaType', com.STDMETHOD(DWORD, POINTER(IMFMediaType))), ('SetCurrentMediaType', com.STDMETHOD(DWORD, POINTER(DWORD), IMFMediaType)), ('SetCurrentPosition', com.STDMETHOD(com.REFIID, POINTER(PROPVARIANT))), ('ReadSample', com.STDMETHOD(DWORD, DWORD, POINTER(DWORD), POINTER(DWORD), POINTER(c_longlong), POINTER(IMFSample))), ('Flush', com.STDMETHOD(DWORD)), ('GetServiceForStream', com.STDMETHOD()), ('GetPresentationAttribute', com.STDMETHOD(DWORD, com.REFIID, POINTER(PROPVARIANT)))]