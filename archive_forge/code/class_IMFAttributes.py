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
class IMFAttributes(com.pIUnknown):
    _methods_ = [('GetItem', com.STDMETHOD()), ('GetItemType', com.STDMETHOD()), ('CompareItem', com.STDMETHOD()), ('Compare', com.STDMETHOD()), ('GetUINT32', com.STDMETHOD(com.REFIID, POINTER(c_uint32))), ('GetUINT64', com.STDMETHOD(com.REFIID, POINTER(c_uint64))), ('GetDouble', com.STDMETHOD()), ('GetGUID', com.STDMETHOD(com.REFIID, POINTER(com.GUID))), ('GetStringLength', com.STDMETHOD()), ('GetString', com.STDMETHOD()), ('GetAllocatedString', com.STDMETHOD()), ('GetBlobSize', com.STDMETHOD()), ('GetBlob', com.STDMETHOD()), ('GetAllocatedBlob', com.STDMETHOD()), ('GetUnknown', com.STDMETHOD()), ('SetItem', com.STDMETHOD()), ('DeleteItem', com.STDMETHOD()), ('DeleteAllItems', com.STDMETHOD()), ('SetUINT32', com.STDMETHOD(com.REFIID, c_uint32)), ('SetUINT64', com.STDMETHOD()), ('SetDouble', com.STDMETHOD()), ('SetGUID', com.STDMETHOD(com.REFIID, com.REFIID)), ('SetString', com.STDMETHOD()), ('SetBlob', com.STDMETHOD()), ('SetUnknown', com.STDMETHOD(com.REFIID, com.pIUnknown)), ('LockStore', com.STDMETHOD()), ('UnlockStore', com.STDMETHOD()), ('GetCount', com.STDMETHOD()), ('GetItemByIndex', com.STDMETHOD()), ('CopyAllItems', com.STDMETHOD(c_void_p))]