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
class IDWriteTextAnalysisSource(com.IUnknown):
    _methods_ = [('GetTextAtPosition', com.STDMETHOD(UINT32, POINTER(c_wchar_p), POINTER(UINT32))), ('GetTextBeforePosition', com.STDMETHOD(UINT32, POINTER(c_wchar_p), POINTER(UINT32))), ('GetParagraphReadingDirection', com.METHOD(DWRITE_READING_DIRECTION)), ('GetLocaleName', com.STDMETHOD(UINT32, POINTER(UINT32), POINTER(c_wchar_p))), ('GetNumberSubstitution', com.STDMETHOD(UINT32, POINTER(UINT32), c_void_p))]