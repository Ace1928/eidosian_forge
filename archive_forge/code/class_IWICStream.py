from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICStream(IStream, com.pIUnknown):
    _methods_ = [('InitializeFromIStream', com.STDMETHOD(IStream)), ('InitializeFromFilename', com.STDMETHOD(LPCWSTR, DWORD)), ('InitializeFromMemory', com.STDMETHOD(POINTER(BYTE), DWORD)), ('InitializeFromIStreamRegion', com.STDMETHOD())]