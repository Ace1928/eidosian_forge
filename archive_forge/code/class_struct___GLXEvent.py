from OpenGL import platform as _p, constant, extensions
from ctypes import *
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
class struct___GLXEvent(Union):
    __slots__ = ['glxpbufferclobber', 'glxbufferswapcomplete', 'pad']