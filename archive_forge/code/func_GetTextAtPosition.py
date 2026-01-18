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
def GetTextAtPosition(self, textPosition, textString, textLength):
    if textPosition >= self._textlength:
        self._no_ptr = c_wchar_p(None)
        textString[0] = self._no_ptr
        textLength[0] = 0
    else:
        ptr = c_wchar_p(self._text[textPosition:])
        self._ptrs.append(ptr)
        textString[0] = ptr
        textLength[0] = self._textlength - textPosition
    return 0