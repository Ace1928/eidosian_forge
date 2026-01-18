import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
@staticmethod
def _get_shader_source(shader_id):
    """Get the shader source from the shader object"""
    source_length = c_int(0)
    glGetShaderiv(shader_id, GL_SHADER_SOURCE_LENGTH, source_length)
    source_str = create_string_buffer(source_length.value)
    glGetShaderSource(shader_id, source_length, None, source_str)
    return source_str.value.decode('utf8')