from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
class FcValue(Structure):
    _fields_ = [('type', FcType), ('u', _FcValueUnion)]