from OpenGL import arrays
from OpenGL.raw.GL._types import GLenum,GLboolean,GLsizei,GLint,GLuint
from OpenGL.raw.osmesa._types import *
from OpenGL.constant import Constant as _C
from OpenGL import platform as _p
import ctypes
@_f
@_p.types(OSMesaContext, GLenum, GLint, GLint, GLint, OSMesaContext)
def OSMesaCreateContextExt(format, depthBits, stencilBits, accumBits, sharelist):
    pass