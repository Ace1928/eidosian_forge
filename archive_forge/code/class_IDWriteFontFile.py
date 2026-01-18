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
class IDWriteFontFile(com.pIUnknown):
    _methods_ = [('GetReferenceKey', com.STDMETHOD(POINTER(c_void_p), POINTER(UINT32))), ('GetLoader', com.STDMETHOD(POINTER(IDWriteFontFileLoader))), ('Analyze', com.STDMETHOD())]