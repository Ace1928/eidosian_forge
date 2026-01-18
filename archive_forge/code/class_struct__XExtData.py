from OpenGL import platform as _p, constant, extensions
from ctypes import *
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
class struct__XExtData(Structure):
    __slots__ = ['number', 'next', 'free_private', 'private_data']