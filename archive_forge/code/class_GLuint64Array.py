import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLuint64Array(ArrayDatatype, ctypes.POINTER(_types.GLuint64)):
    """Array datatype for GLuint types"""
    baseType = _types.GLuint64
    typeConstant = _types.GL_UNSIGNED_INT64