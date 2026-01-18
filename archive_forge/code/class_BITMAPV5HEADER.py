import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class BITMAPV5HEADER(Structure):
    _fields_ = [('bV5Size', DWORD), ('bV5Width', LONG), ('bV5Height', LONG), ('bV5Planes', WORD), ('bV5BitCount', WORD), ('bV5Compression', DWORD), ('bV5SizeImage', DWORD), ('bV5XPelsPerMeter', LONG), ('bV5YPelsPerMeter', LONG), ('bV5ClrUsed', DWORD), ('bV5ClrImportant', DWORD), ('bV5RedMask', DWORD), ('bV5GreenMask', DWORD), ('bV5BlueMask', DWORD), ('bV5AlphaMask', DWORD), ('bV5CSType', DWORD), ('bV5Endpoints', CIEXYZTRIPLE), ('bV5GammaRed', DWORD), ('bV5GammaGreen', DWORD), ('bV5GammaBlue', DWORD), ('bV5Intent', DWORD), ('bV5ProfileData', DWORD), ('bV5ProfileSize', DWORD), ('bV5Reserved', DWORD)]