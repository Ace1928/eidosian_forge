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
class DWRITE_GLYPH_OFFSET(ctypes.Structure):
    _fields_ = (('advanceOffset', FLOAT), ('ascenderOffset', FLOAT))

    def __repr__(self):
        return f'DWRITE_GLYPH_OFFSET({self.advanceOffset}, {self.ascenderOffset})'