from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICBitmapFrameDecode(IWICBitmapSource, com.pIUnknown):
    _methods_ = [('GetMetadataQueryReader', com.STDMETHOD(POINTER(IWICMetadataQueryReader))), ('GetColorContexts', com.STDMETHOD()), ('GetThumbnail', com.STDMETHOD(POINTER(IWICBitmapSource)))]