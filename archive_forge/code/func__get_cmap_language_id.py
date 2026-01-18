import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_cmap_language_id(self):
    return FT_Get_CMap_Language_ID(self._FT_Charmap)