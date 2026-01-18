from the GLUT module.  Note that any other implementation that also provides
from OpenGL import platform, arrays
from OpenGL import constant
from OpenGL.GLUT import special
from OpenGL import wrapper as _wrapper
from OpenGL.raw.GL._types import *
import ctypes
FreeGLUT extensions to the GLUT API

This module will provide the FreeGLUT extensions if they are available
from the GLUT module.  Note that any other implementation that also provides
these entry points will also retrieve the entry points with this module.
