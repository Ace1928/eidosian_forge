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
class IDWriteFontFamily(IDWriteFontList, com.pIUnknown):
    _methods_ = [('GetFamilyNames', com.STDMETHOD(POINTER(IDWriteLocalizedStrings))), ('GetFirstMatchingFont', com.STDMETHOD(DWRITE_FONT_WEIGHT, DWRITE_FONT_STRETCH, DWRITE_FONT_STYLE, c_void_p)), ('GetMatchingFonts', com.STDMETHOD())]