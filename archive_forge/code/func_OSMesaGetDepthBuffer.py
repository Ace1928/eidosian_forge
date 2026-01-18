from OpenGL import arrays
from OpenGL.raw.GL._types import GLenum,GLboolean,GLsizei,GLint,GLuint
from OpenGL.raw.osmesa._types import *
from OpenGL.constant import Constant as _C
from OpenGL import platform as _p
import ctypes
def OSMesaGetDepthBuffer(c):
    width, height, bytesPerValue = (GLint(), GLint(), GLint())
    buffer = ctypes.POINTER(GLint)()
    if _p.PLATFORM.GL.OSMesaGetDepthBuffer(c, ctypes.byref(width), ctypes.byref(height), ctypes.byref(bytesPerValue), ctypes.byref(buffer)):
        return (width.value, height.value, bytesPerValue.value, buffer)
    else:
        return (0, 0, 0, None)