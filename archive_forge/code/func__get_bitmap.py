import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_bitmap(self):
    return Bitmap(self._FT_GlyphSlot.contents.bitmap)