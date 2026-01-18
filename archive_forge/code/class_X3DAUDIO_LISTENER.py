import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class X3DAUDIO_LISTENER(Structure):
    _fields_ = [('OrientFront', X3DAUDIO_VECTOR), ('OrientTop', X3DAUDIO_VECTOR), ('Position', X3DAUDIO_VECTOR), ('Velocity', X3DAUDIO_VECTOR), ('pCone', POINTER(X3DAUDIO_CONE))]