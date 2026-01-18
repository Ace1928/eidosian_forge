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
class IDWriteFactory3(IDWriteFactory2, com.pIUnknown):
    _methods_ = [('CreateGlyphRunAnalysis', com.STDMETHOD()), ('CreateCustomRenderingParams3', com.STDMETHOD()), ('CreateFontFaceReference', com.STDMETHOD()), ('CreateFontFaceReference', com.STDMETHOD()), ('GetSystemFontSet', com.STDMETHOD()), ('CreateFontSetBuilder', com.STDMETHOD(POINTER(IDWriteFontSetBuilder))), ('CreateFontCollectionFromFontSet', com.STDMETHOD(IDWriteFontSet, POINTER(IDWriteFontCollection1))), ('GetSystemFontCollection3', com.STDMETHOD()), ('GetFontDownloadQueue', com.STDMETHOD())]