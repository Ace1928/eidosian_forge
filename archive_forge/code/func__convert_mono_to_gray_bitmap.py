import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _convert_mono_to_gray_bitmap(self):
    bitmap_data = cast(self._bitmap.buffer, POINTER(c_ubyte * (self._pitch * self._height))).contents
    data = (c_ubyte * (self._pitch * 8 * self._height))()
    data_i = 0
    for byte in bitmap_data:
        data[data_i + 0] = byte & 128 and 255 or 0
        data[data_i + 1] = byte & 64 and 255 or 0
        data[data_i + 2] = byte & 32 and 255 or 0
        data[data_i + 3] = byte & 16 and 255 or 0
        data[data_i + 4] = byte & 8 and 255 or 0
        data[data_i + 5] = byte & 4 and 255 or 0
        data[data_i + 6] = byte & 2 and 255 or 0
        data[data_i + 7] = byte & 1 and 255 or 0
        data_i += 8
    self._data = data
    self._pitch <<= 3