import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLfloat16Array(ArrayDatatype, ctypes.POINTER(_types.GLushort)):
    """Array datatype for float16 as GLushort types"""
    baseType = _types.GLushort
    typeConstant = _types.GL_HALF_FLOAT