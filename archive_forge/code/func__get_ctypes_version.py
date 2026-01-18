import ctypes
from OpenGL.constant import Constant
from OpenGL._bytes import bytes,unicode,as_8_bit, long
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
from OpenGL.platform import PLATFORM as _p
def _get_ctypes_version():
    return [int(i) for i in ctypes.__version__.split('.')[:3]]