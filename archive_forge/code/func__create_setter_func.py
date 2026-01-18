import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
@staticmethod
def _create_setter_func(program, location, gl_setter, c_array, length, ptr, is_matrix, dsa):
    """Factory function for creating simplified Uniform setters"""
    if dsa:
        if is_matrix:

            def setter_func(value):
                c_array[:] = value
                gl_setter(program, location, 1, GL_FALSE, ptr)
        elif length == 1:

            def setter_func(value):
                c_array[0] = value
                gl_setter(program, location, 1, ptr)
        elif length > 1:

            def setter_func(values):
                c_array[:] = values
                gl_setter(program, location, 1, ptr)
        else:
            raise ShaderException('Uniform type not yet supported.')
        return setter_func
    else:
        if is_matrix:

            def setter_func(value):
                glUseProgram(program)
                c_array[:] = value
                gl_setter(location, 1, GL_FALSE, ptr)
        elif length == 1:

            def setter_func(value):
                glUseProgram(program)
                c_array[0] = value
                gl_setter(location, 1, ptr)
        elif length > 1:

            def setter_func(values):
                glUseProgram(program)
                c_array[:] = values
                gl_setter(location, 1, ptr)
        else:
            raise ShaderException('Uniform type not yet supported.')
        return setter_func