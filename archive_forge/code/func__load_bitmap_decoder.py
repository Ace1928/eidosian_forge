from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
def _load_bitmap_decoder(self, filename, file):
    data = file.read()
    hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    ptr = kernel32.GlobalLock(hglob)
    memmove(ptr, data, len(data))
    kernel32.GlobalUnlock(hglob)
    stream = com.pIUnknown()
    ole32.CreateStreamOnHGlobal(hglob, True, byref(stream))
    decoder = IWICBitmapDecoder()
    status = self._factory.CreateDecoderFromStream(stream, None, WICDecodeMetadataCacheOnDemand, byref(decoder))
    if status != 0:
        stream.Release()
        raise ImageDecodeException('WIC cannot load %r' % (filename or file))
    return (decoder, stream)