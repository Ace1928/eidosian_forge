from ctypes import *
import warnings
from pyglet.gl.lib import MissingFunctionException
from pyglet.gl.gl import *
from pyglet.gl import gl_info
from pyglet.gl.wgl import *
from pyglet.gl.wglext_arb import *
from pyglet.util import asstr
class WGLInfoException(Exception):
    pass