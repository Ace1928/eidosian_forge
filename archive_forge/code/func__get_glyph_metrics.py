import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _get_glyph_metrics(self):
    self._width = self._glyph_slot.bitmap.width
    self._height = self._glyph_slot.bitmap.rows
    self._mode = self._glyph_slot.bitmap.pixel_mode
    self._pitch = self._glyph_slot.bitmap.pitch
    self._baseline = self._height - self._glyph_slot.bitmap_top
    self._lsb = self._glyph_slot.bitmap_left
    self._advance_x = int(f26p6_to_float(self._glyph_slot.advance.x))