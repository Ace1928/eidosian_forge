from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICBitmapDecoder(com.pIUnknown):
    _methods_ = [('QueryCapability', com.STDMETHOD()), ('Initialize', com.STDMETHOD()), ('GetContainerFormat', com.STDMETHOD()), ('GetDecoderInfo', com.STDMETHOD()), ('CopyPalette', com.STDMETHOD()), ('GetMetadataQueryReader', com.STDMETHOD(POINTER(IWICMetadataQueryReader))), ('GetPreview', com.STDMETHOD()), ('GetColorContexts', com.STDMETHOD()), ('GetThumbnail', com.STDMETHOD()), ('GetFrameCount', com.STDMETHOD(POINTER(UINT))), ('GetFrame', com.STDMETHOD(UINT, POINTER(IWICBitmapFrameDecode)))]