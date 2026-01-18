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
class IDWriteTextAnalysisSink(com.IUnknown):
    _methods_ = [('SetScriptAnalysis', com.STDMETHOD(UINT32, UINT32, POINTER(DWRITE_SCRIPT_ANALYSIS))), ('SetLineBreakpoints', com.STDMETHOD(UINT32, UINT32, c_void_p)), ('SetBidiLevel', com.STDMETHOD(UINT32, UINT32, UINT8, UINT8)), ('SetNumberSubstitution', com.STDMETHOD(UINT32, UINT32, c_void_p))]