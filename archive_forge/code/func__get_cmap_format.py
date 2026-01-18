import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_cmap_format(self):
    return FT_Get_CMap_Format(self._FT_Charmap)