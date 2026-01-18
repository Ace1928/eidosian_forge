import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2_EFFECT_CHAIN(ctypes.Structure):
    _fields_ = [('EffectCount', UINT32), ('pEffectDescriptors', POINTER(XAUDIO2_EFFECT_DESCRIPTOR))]