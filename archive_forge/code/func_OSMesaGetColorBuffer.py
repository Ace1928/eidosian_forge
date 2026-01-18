from OpenGL import arrays
from OpenGL.raw.GL._types import GLenum,GLboolean,GLsizei,GLint,GLuint
from OpenGL.raw.osmesa._types import *
from OpenGL.constant import Constant as _C
from OpenGL import platform as _p
import ctypes
def OSMesaGetColorBuffer(c):
    width, height, format = (GLint(), GLint(), GLint())
    buffer = ctypes.c_void_p()
    if _p.PLATFORM.GL.OSMesaGetColorBuffer(c, ctypes.byref(width), ctypes.byref(height), ctypes.byref(format), ctypes.byref(buffer)):
        return (width.value, height.value, format.value, buffer)
    else:
        return (0, 0, 0, None)