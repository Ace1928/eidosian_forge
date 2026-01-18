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
class IDWriteFactory(com.pIUnknown):
    _methods_ = [('GetSystemFontCollection', com.STDMETHOD(POINTER(IDWriteFontCollection), BOOL)), ('CreateCustomFontCollection', com.STDMETHOD(POINTER(IDWriteFontCollectionLoader), c_void_p, UINT32, POINTER(IDWriteFontCollection))), ('RegisterFontCollectionLoader', com.STDMETHOD(POINTER(IDWriteFontCollectionLoader))), ('UnregisterFontCollectionLoader', com.STDMETHOD(POINTER(IDWriteFontCollectionLoader))), ('CreateFontFileReference', com.STDMETHOD(c_wchar_p, c_void_p, POINTER(IDWriteFontFile))), ('CreateCustomFontFileReference', com.STDMETHOD(c_void_p, UINT32, POINTER(IDWriteFontFileLoader_LI), POINTER(IDWriteFontFile))), ('CreateFontFace', com.STDMETHOD()), ('CreateRenderingParams', com.STDMETHOD(POINTER(IDWriteRenderingParams))), ('CreateMonitorRenderingParams', com.STDMETHOD()), ('CreateCustomRenderingParams', com.STDMETHOD(FLOAT, FLOAT, FLOAT, UINT, UINT, POINTER(IDWriteRenderingParams))), ('RegisterFontFileLoader', com.STDMETHOD(c_void_p)), ('UnregisterFontFileLoader', com.STDMETHOD(POINTER(IDWriteFontFileLoader_LI))), ('CreateTextFormat', com.STDMETHOD(c_wchar_p, IDWriteFontCollection, DWRITE_FONT_WEIGHT, DWRITE_FONT_STYLE, DWRITE_FONT_STRETCH, FLOAT, c_wchar_p, POINTER(IDWriteTextFormat))), ('CreateTypography', com.STDMETHOD(POINTER(IDWriteTypography))), ('GetGdiInterop', com.STDMETHOD(POINTER(IDWriteGdiInterop))), ('CreateTextLayout', com.STDMETHOD(c_wchar_p, UINT32, IDWriteTextFormat, FLOAT, FLOAT, POINTER(IDWriteTextLayout))), ('CreateGdiCompatibleTextLayout', com.STDMETHOD()), ('CreateEllipsisTrimmingSign', com.STDMETHOD()), ('CreateTextAnalyzer', com.STDMETHOD(POINTER(IDWriteTextAnalyzer))), ('CreateNumberSubstitution', com.STDMETHOD()), ('CreateGlyphRunAnalysis', com.STDMETHOD())]