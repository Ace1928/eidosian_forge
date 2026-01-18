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
class IDWriteFontFace(com.pIUnknown):
    _methods_ = [('GetType', com.STDMETHOD()), ('GetFiles', com.STDMETHOD(POINTER(UINT32), POINTER(IDWriteFontFile))), ('GetIndex', com.STDMETHOD()), ('GetSimulations', com.STDMETHOD()), ('IsSymbolFont', com.STDMETHOD()), ('GetMetrics', com.METHOD(c_void, POINTER(DWRITE_FONT_METRICS))), ('GetGlyphCount', com.METHOD(UINT16)), ('GetDesignGlyphMetrics', com.STDMETHOD(POINTER(UINT16), UINT32, POINTER(DWRITE_GLYPH_METRICS), BOOL)), ('GetGlyphIndices', com.STDMETHOD(POINTER(UINT32), UINT32, POINTER(UINT16))), ('TryGetFontTable', com.STDMETHOD(UINT32, c_void_p, POINTER(UINT32), c_void_p, POINTER(BOOL))), ('ReleaseFontTable', com.METHOD(c_void)), ('GetGlyphRunOutline', com.STDMETHOD()), ('GetRecommendedRenderingMode', com.STDMETHOD()), ('GetGdiCompatibleMetrics', com.STDMETHOD()), ('GetGdiCompatibleGlyphMetrics', com.STDMETHOD())]