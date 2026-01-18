from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
class IPropertyStore(com.pIUnknown):
    _methods_ = [('GetCount', com.STDMETHOD(POINTER(DWORD))), ('GetAt', com.STDMETHOD(DWORD, POINTER(PROPERTYKEY))), ('GetValue', com.STDMETHOD(REFPROPERTYKEY, POINTER(PROPVARIANT))), ('SetValue', com.STDMETHOD()), ('Commit', com.STDMETHOD())]