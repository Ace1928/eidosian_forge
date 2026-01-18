from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.EXT.device_base import *
from OpenGL.raw.EGL.EXT.device_base import _EXTENSION_NAME
Utility function that retrieves platform devices