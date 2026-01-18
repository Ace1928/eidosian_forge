from OpenGL.raw import GLU as _simple
from OpenGL import platform, converters, wrapper
from OpenGL.GLU import glustruct
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL import arrays, error
import ctypes
import weakref
from OpenGL.platform import PLATFORM
import OpenGL
from OpenGL import _configflags
def _callbackWithType(funcType):
    """Get gluNurbsCallback function with set last arg-type"""
    result = platform.copyBaseFunction(_simple.gluNurbsCallback)
    result.argtypes = [ctypes.POINTER(GLUnurbs), _simple.GLenum, funcType]
    assert result.argtypes[-1] == funcType
    return result