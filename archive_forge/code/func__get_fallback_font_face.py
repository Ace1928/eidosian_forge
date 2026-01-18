import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
def _get_fallback_font_face(self, text_index, text_length):
    if WINDOWS_8_1_OR_GREATER:
        out_length = UINT32()
        fb_font = IDWriteFont()
        scale = FLOAT()
        self._fallback.MapCharacters(self._glyph_renderer._text_analysis, text_index, text_length, None, None, self._weight, self._style, self._stretch, byref(out_length), byref(fb_font), byref(scale))
        if fb_font:
            fb_font_face = IDWriteFontFace()
            fb_font.CreateFontFace(byref(fb_font_face))
            return fb_font_face
    return None