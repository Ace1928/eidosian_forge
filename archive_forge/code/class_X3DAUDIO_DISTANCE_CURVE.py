import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class X3DAUDIO_DISTANCE_CURVE(ctypes.Structure):
    _fields_ = [('pPoints', POINTER(X3DAUDIO_DISTANCE_CURVE_POINT)), ('PointCount', UINT32)]