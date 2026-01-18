from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
class IMMDevice(com.pIUnknown):
    _methods_ = [('Activate', com.STDMETHOD(com.REFIID, DWORD, POINTER(PROPVARIANT))), ('OpenPropertyStore', com.STDMETHOD(UINT, POINTER(IPropertyStore))), ('GetId', com.STDMETHOD(POINTER(LPWSTR))), ('GetState', com.STDMETHOD(POINTER(DWORD)))]