import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
class BITMAPV4HEADER(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [('biSize', DWORD), ('biWidth', LONG), ('biHeight', LONG), ('biPlanes', WORD), ('biBitCount', WORD), ('biCompression', DWORD), ('biSizeImage', DWORD), ('biXPelsPerMeter', LONG), ('biYPelsPerMeter', LONG), ('biClrUsed', DWORD), ('biClrImportant', DWORD), ('bV4RedMask', DWORD), ('bV4GreenMask', DWORD), ('bV4BlueMask', DWORD), ('bV4AlphaMask', DWORD), ('bV4CSType', DWORD), ('bV4Endpoints', CIEXYZTRIPLE), ('bV4GammaRed', DWORD), ('bV4GammaGreen', DWORD), ('bV4GammaBlue', DWORD)]