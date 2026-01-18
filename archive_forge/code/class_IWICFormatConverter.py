from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICFormatConverter(IWICBitmapSource, com.pIUnknown):
    _methods_ = [('Initialize', com.STDMETHOD(IWICBitmapSource, REFWICPixelFormatGUID, WICBitmapDitherType, c_void_p, DOUBLE, WICBitmapPaletteType)), ('CanConvert', com.STDMETHOD(REFWICPixelFormatGUID, REFWICPixelFormatGUID, POINTER(BOOL)))]