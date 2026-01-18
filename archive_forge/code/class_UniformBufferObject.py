import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
class UniformBufferObject:
    __slots__ = ('buffer', 'view', '_view_ptr', 'index')

    def __init__(self, view_class, buffer_size, index):
        self.buffer = BufferObject(buffer_size)
        self.view = view_class()
        self._view_ptr = pointer(self.view)
        self.index = index

    @property
    def id(self):
        return self.buffer.id

    def bind(self, index=None):
        glBindBufferBase(GL_UNIFORM_BUFFER, self.index if index is None else index, self.buffer.id)

    def read(self):
        """Read the byte contents of the buffer"""
        glBindBuffer(GL_ARRAY_BUFFER, self.buffer.id)
        ptr = glMapBufferRange(GL_ARRAY_BUFFER, 0, self.buffer.size, GL_MAP_READ_BIT)
        data = string_at(ptr, size=self.buffer.size)
        glUnmapBuffer(GL_ARRAY_BUFFER)
        return data

    def __enter__(self):
        return self.view

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bind()
        self.buffer.set_data(self._view_ptr)

    def __repr__(self):
        return '{0}(id={1})'.format(self.__class__.__name__, self.buffer.id)