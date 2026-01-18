import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
@staticmethod
def _get_tuple(parameter: int):
    val_x = GLint()
    val_y = GLint()
    val_z = GLint()
    for i, value in enumerate((val_x, val_y, val_z)):
        glGetIntegeri_v(parameter, i, byref(value))
    return (val_x.value, val_y.value, val_z.value)