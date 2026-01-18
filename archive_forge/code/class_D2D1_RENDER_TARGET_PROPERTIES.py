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
class D2D1_RENDER_TARGET_PROPERTIES(Structure):
    _fields_ = (('type', D2D1_RENDER_TARGET_TYPE), ('pixelFormat', D2D1_PIXEL_FORMAT), ('dpiX', FLOAT), ('dpiY', FLOAT), ('usage', D2D1_RENDER_TARGET_USAGE), ('minLevel', D2D1_FEATURE_LEVEL))