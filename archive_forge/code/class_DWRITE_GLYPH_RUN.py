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
class DWRITE_GLYPH_RUN(ctypes.Structure):
    _fields_ = (('fontFace', IDWriteFontFace), ('fontEmSize', FLOAT), ('glyphCount', UINT32), ('glyphIndices', POINTER(UINT16)), ('glyphAdvances', POINTER(FLOAT)), ('glyphOffsets', POINTER(DWRITE_GLYPH_OFFSET)), ('isSideways', BOOL), ('bidiLevel', UINT32))