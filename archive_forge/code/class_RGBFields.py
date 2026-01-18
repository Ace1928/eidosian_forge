import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
class RGBFields(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [('red', DWORD), ('green', DWORD), ('blue', DWORD)]