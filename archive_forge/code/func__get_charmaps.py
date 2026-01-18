import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_charmaps(self):
    charmaps = []
    n = self._FT_Face.contents.num_charmaps
    FT_charmaps = self._FT_Face.contents.charmaps
    for i in range(n):
        charmaps.append(Charmap(FT_charmaps[i]))
    return charmaps