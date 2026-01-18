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
def _vec3(self, function, size=3):
    """Convert first arg to size-element array, do OOR on arg2 if present"""

    def vec(*args):
        vec = self.ptrAsArray(args[0], size, arrays.GLfloatArray)
        if len(args) > 1:
            oor = self.originalObject(args[1])
            return function(vec, oor)
        else:
            return function(vec)
    return vec