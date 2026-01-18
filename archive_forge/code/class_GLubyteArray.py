import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class GLubyteArray(ArrayDatatype, ctypes.POINTER(_types.GLubyte)):
    """Array datatype for GLubyte types"""
    baseType = _types.GLubyte
    typeConstant = _types.GL_UNSIGNED_BYTE