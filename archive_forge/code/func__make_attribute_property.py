import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
def _make_attribute_property(name):

    def _attribute_getter(self):
        attribute = self.domain.attribute_names[name]
        region = attribute.buffer.get_region(self.start, self.count)
        attribute.buffer.invalidate_region(self.start, self.count)
        return region

    def _attribute_setter(self, data):
        attribute = self.domain.attribute_names[name]
        attribute.buffer.set_region(self.start, self.count, data)
    return property(_attribute_getter, _attribute_setter)