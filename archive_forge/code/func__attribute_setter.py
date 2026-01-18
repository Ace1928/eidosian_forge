import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
def _attribute_setter(self, data):
    attribute = self.domain.attribute_names[name]
    attribute.buffer.set_region(self.start, self.count, data)