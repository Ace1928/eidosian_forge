import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_charmap(self):
    return Charmap(self._FT_Face.contents.charmap)