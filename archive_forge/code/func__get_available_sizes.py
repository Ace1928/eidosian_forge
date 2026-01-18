import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_available_sizes(self):
    sizes = []
    n = self.num_fixed_sizes
    FT_sizes = self._FT_Face.contents.available_sizes
    for i in range(n):
        sizes.append(BitmapSize(FT_sizes[i]))
    return sizes