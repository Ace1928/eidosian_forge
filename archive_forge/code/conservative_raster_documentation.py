from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.NV.conservative_raster import *
from OpenGL.raw.GLES2.NV.conservative_raster import _EXTENSION_NAME
Return boolean indicating whether this extension is available