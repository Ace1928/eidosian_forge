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
class IDWriteFontFace1(IDWriteFontFace, com.pIUnknown):
    _methods_ = [('GetMetric1', com.STDMETHOD()), ('GetGdiCompatibleMetrics1', com.STDMETHOD()), ('GetCaretMetrics', com.STDMETHOD()), ('GetUnicodeRanges', com.STDMETHOD()), ('IsMonospacedFont', com.STDMETHOD()), ('GetDesignGlyphAdvances', com.METHOD(c_void, POINTER(DWRITE_FONT_METRICS))), ('GetGdiCompatibleGlyphAdvances', com.STDMETHOD()), ('GetKerningPairAdjustments', com.STDMETHOD(UINT32, POINTER(UINT16), POINTER(INT32))), ('HasKerningPairs', com.METHOD(BOOL)), ('GetRecommendedRenderingMode1', com.STDMETHOD()), ('GetVerticalGlyphVariants', com.STDMETHOD()), ('HasVerticalGlyphVariants', com.STDMETHOD())]