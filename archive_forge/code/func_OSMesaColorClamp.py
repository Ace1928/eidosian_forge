from OpenGL import arrays
from OpenGL.raw.GL._types import GLenum,GLboolean,GLsizei,GLint,GLuint
from OpenGL.raw.osmesa._types import *
from OpenGL.constant import Constant as _C
from OpenGL import platform as _p
import ctypes
@_f
@_p.types(GLboolean)
def OSMesaColorClamp(enable):
    """Enable/disable color clamping, off by default

    New in Mesa 6.4.2
    """