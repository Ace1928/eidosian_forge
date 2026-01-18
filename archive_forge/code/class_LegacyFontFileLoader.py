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
class LegacyFontFileLoader(com.COMObject):
    _interfaces_ = [IDWriteFontFileLoader_LI]

    def __init__(self):
        super().__init__()
        self._streams = {}

    def CreateStreamFromKey(self, fontfileReferenceKey, fontFileReferenceKeySize, fontFileStream):
        convert_index = cast(fontfileReferenceKey, POINTER(c_uint32))
        self._ptr = ctypes.cast(self._streams[convert_index.contents.value].as_interface(IDWriteFontFileStream), POINTER(IDWriteFontFileStream))
        fontFileStream[0] = self._ptr
        return 0

    def SetCurrentFont(self, index, data):
        self._streams[index] = MyFontFileStream(data)