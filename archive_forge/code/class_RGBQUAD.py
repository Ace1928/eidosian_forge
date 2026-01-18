import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
class RGBQUAD(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [('rgbBlue', BYTE), ('rgbGreen', BYTE), ('rgbRed', BYTE), ('rgbReserved', BYTE)]

    def __repr__(self):
        return '<%d, %d, %d>' % (self.rgbRed, self.rgbGreen, self.rgbBlue)