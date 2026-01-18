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
class IDWriteTextFormat(com.pIUnknown):
    _methods_ = [('SetTextAlignment', com.STDMETHOD(DWRITE_TEXT_ALIGNMENT)), ('SetParagraphAlignment', com.STDMETHOD()), ('SetWordWrapping', com.STDMETHOD()), ('SetReadingDirection', com.STDMETHOD()), ('SetFlowDirection', com.STDMETHOD()), ('SetIncrementalTabStop', com.STDMETHOD()), ('SetTrimming', com.STDMETHOD()), ('SetLineSpacing', com.STDMETHOD()), ('GetTextAlignment', com.STDMETHOD()), ('GetParagraphAlignment', com.STDMETHOD()), ('GetWordWrapping', com.STDMETHOD()), ('GetReadingDirection', com.STDMETHOD()), ('GetFlowDirection', com.STDMETHOD()), ('GetIncrementalTabStop', com.STDMETHOD()), ('GetTrimming', com.STDMETHOD()), ('GetLineSpacing', com.STDMETHOD()), ('GetFontCollection', com.STDMETHOD()), ('GetFontFamilyNameLength', com.STDMETHOD(UINT32, POINTER(UINT32))), ('GetFontFamilyName', com.STDMETHOD(UINT32, c_wchar_p, UINT32)), ('GetFontWeight', com.STDMETHOD()), ('GetFontStyle', com.STDMETHOD()), ('GetFontStretch', com.STDMETHOD()), ('GetFontSize', com.STDMETHOD()), ('GetLocaleNameLength', com.STDMETHOD()), ('GetLocaleName', com.STDMETHOD())]