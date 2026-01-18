import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
class ShaderException(BaseException):
    pass