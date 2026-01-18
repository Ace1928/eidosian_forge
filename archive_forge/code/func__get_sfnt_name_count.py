import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_sfnt_name_count(self):
    return FT_Get_Sfnt_Name_Count(self._FT_Face)