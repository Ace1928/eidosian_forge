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
class IDWriteTextLayout(IDWriteTextFormat, com.pIUnknown):
    _methods_ = [('SetMaxWidth', com.STDMETHOD()), ('SetMaxHeight', com.STDMETHOD()), ('SetFontCollection', com.STDMETHOD()), ('SetFontFamilyName', com.STDMETHOD()), ('SetFontWeight', com.STDMETHOD()), ('SetFontStyle', com.STDMETHOD()), ('SetFontStretch', com.STDMETHOD()), ('SetFontSize', com.STDMETHOD()), ('SetUnderline', com.STDMETHOD()), ('SetStrikethrough', com.STDMETHOD()), ('SetDrawingEffect', com.STDMETHOD()), ('SetInlineObject', com.STDMETHOD()), ('SetTypography', com.STDMETHOD(IDWriteTypography, DWRITE_TEXT_RANGE)), ('SetLocaleName', com.STDMETHOD()), ('GetMaxWidth', com.METHOD(FLOAT)), ('GetMaxHeight', com.METHOD(FLOAT)), ('GetFontCollection2', com.STDMETHOD()), ('GetFontFamilyNameLength2', com.STDMETHOD(UINT32, POINTER(UINT32), c_void_p)), ('GetFontFamilyName2', com.STDMETHOD(UINT32, c_wchar_p, UINT32, c_void_p)), ('GetFontWeight2', com.STDMETHOD(UINT32, POINTER(DWRITE_FONT_WEIGHT), POINTER(DWRITE_TEXT_RANGE))), ('GetFontStyle2', com.STDMETHOD()), ('GetFontStretch2', com.STDMETHOD()), ('GetFontSize2', com.STDMETHOD()), ('GetUnderline', com.STDMETHOD()), ('GetStrikethrough', com.STDMETHOD(UINT32, POINTER(BOOL), POINTER(DWRITE_TEXT_RANGE))), ('GetDrawingEffect', com.STDMETHOD()), ('GetInlineObject', com.STDMETHOD()), ('GetTypography', com.STDMETHOD(UINT32, POINTER(IDWriteTypography), POINTER(DWRITE_TEXT_RANGE))), ('GetLocaleNameLength1', com.STDMETHOD()), ('GetLocaleName1', com.STDMETHOD()), ('Draw', com.STDMETHOD()), ('GetLineMetrics', com.STDMETHOD()), ('GetMetrics', com.STDMETHOD(POINTER(DWRITE_TEXT_METRICS))), ('GetOverhangMetrics', com.STDMETHOD(POINTER(DWRITE_OVERHANG_METRICS))), ('GetClusterMetrics', com.STDMETHOD(POINTER(DWRITE_CLUSTER_METRICS), UINT32, POINTER(UINT32))), ('DetermineMinWidth', com.STDMETHOD(POINTER(FLOAT))), ('HitTestPoint', com.STDMETHOD()), ('HitTestTextPosition', com.STDMETHOD()), ('HitTestTextRange', com.STDMETHOD())]