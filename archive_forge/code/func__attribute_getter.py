import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
def _attribute_getter(self):
    attribute = self.domain.attribute_names[name]
    region = attribute.buffer.get_region(self.start, self.count)
    attribute.buffer.invalidate_region(self.start, self.count)
    return region