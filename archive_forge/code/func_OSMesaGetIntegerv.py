from OpenGL import arrays
from OpenGL.raw.GL._types import GLenum,GLboolean,GLsizei,GLint,GLuint
from OpenGL.raw.osmesa._types import *
from OpenGL.constant import Constant as _C
from OpenGL import platform as _p
import ctypes
def OSMesaGetIntegerv(pname):
    value = GLint()
    _p.PLATFORM.GL.OSMesaGetIntegerv(pname, ctypes.byref(value))
    return value.value