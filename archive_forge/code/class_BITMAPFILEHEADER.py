import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
class BITMAPFILEHEADER(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [('bfType', WORD), ('bfSize', DWORD), ('bfReserved1', WORD), ('bfReserved2', WORD), ('bfOffBits', DWORD)]