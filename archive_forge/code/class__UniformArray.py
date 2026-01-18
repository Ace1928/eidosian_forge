import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
class _UniformArray:
    """Wrapper of the GLSL array data inside a Uniform.
    Allows access to get and set items for a more Pythonic implementation.
    Types with a length longer than 1 will be returned as tuples as an inner list would not support individual value
    reassignment. Array data must either be set in full, or by indexing."""
    __slots__ = ('_uniform', '_gl_type', '_gl_getter', '_gl_setter', '_is_matrix', '_dsa', '_c_array', '_ptr')

    def __init__(self, uniform, gl_getter, gl_setter, gl_type, is_matrix, dsa):
        self._uniform = uniform
        self._gl_type = gl_type
        self._gl_getter = gl_getter
        self._gl_setter = gl_setter
        self._is_matrix = is_matrix
        self._dsa = dsa
        if self._uniform.length > 1:
            self._c_array = (gl_type * self._uniform.length * self._uniform.size)()
        else:
            self._c_array = (gl_type * self._uniform.size)()
        self._ptr = cast(self._c_array, POINTER(gl_type))

    def __len__(self):
        return self._uniform.size

    def __delitem__(self, key):
        raise ShaderException('Deleting items is not support for UniformArrays.')

    def __getitem__(self, key):
        if isinstance(key, slice):
            sliced_data = self._c_array[key]
            if self._uniform.length > 1:
                return [tuple(data) for data in sliced_data]
            else:
                return tuple([data for data in sliced_data])
        value = self._c_array[key]
        return tuple(value) if self._uniform.length > 1 else value

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self._c_array[key] = value
            self._update_uniform(self._ptr)
            return
        self._c_array[key] = value
        if self._uniform.length > 1:
            assert len(value) == self._uniform.length, f'Setting this key requires {self._uniform.length} values, received {len(value)}.'
            data = (self._gl_type * self._uniform.length)(*value)
        else:
            data = self._gl_type(value)
        self._update_uniform(data, offset=key)

    def get(self):
        self._gl_getter(self._uniform.program, self._uniform.location, self._ptr)
        return self

    def set(self, values):
        assert len(self._c_array) == len(values), f'Size of data ({len(values)}) does not match size of the uniform: {len(self._c_array)}.'
        self._c_array[:] = values
        self._update_uniform(self._ptr)

    def _update_uniform(self, data, offset=0):
        if offset != 0:
            size = 1
        else:
            size = self._uniform.size
        if self._dsa:
            self._gl_setter(self._uniform.program, self._uniform.location + offset, size, data)
        else:
            glUseProgram(self._uniform.program)
            self._gl_setter(self._uniform.location + offset, size, data)

    def __repr__(self):
        data = [tuple(data) if self._uniform.length > 1 else data for data in self._c_array]
        return f'UniformArray(uniform={self._uniform.name}, data={data})'