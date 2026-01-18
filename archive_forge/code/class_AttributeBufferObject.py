import sys
import ctypes
from functools import lru_cache
import pyglet
from pyglet.gl import *
class AttributeBufferObject(BufferObject):
    """A buffer with system-memory backed store.

    Updates to the data via `set_data` and `set_data_region` will be held
    in local memory until `buffer_data` is called.  The advantage is that
    fewer OpenGL calls are needed, which can increasing performance at the
    expense of system memory.
    """

    def __init__(self, size, attribute, usage=GL_DYNAMIC_DRAW):
        super().__init__(size, usage)
        number = size // attribute.element_size
        self.data = (attribute.c_type * number)()
        self.data_ptr = ctypes.addressof(self.data)
        self._dirty_min = sys.maxsize
        self._dirty_max = 0
        self._dirty = False
        self.attribute_stride = attribute.stride
        self.attribute_count = attribute.count
        self.attribute_ctype = attribute.c_type

    def sub_data(self):
        """Updates the buffer if any data has been changed or invalidated. Allows submitting multiple changes at once,
        rather than having to call glBufferSubData for every change."""
        if not self._dirty:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        size = self._dirty_max - self._dirty_min
        if size > 0:
            if size == self.size:
                glBufferData(GL_ARRAY_BUFFER, self.size, self.data, self.usage)
            else:
                glBufferSubData(GL_ARRAY_BUFFER, self._dirty_min, size, self.data_ptr + self._dirty_min)
            self._dirty_min = sys.maxsize
            self._dirty_max = 0
            self._dirty = False

    @lru_cache(maxsize=None)
    def get_region(self, start, count):
        byte_start = self.attribute_stride * start
        array_count = self.attribute_count * count
        ptr_type = ctypes.POINTER(self.attribute_ctype * array_count)
        return ctypes.cast(self.data_ptr + byte_start, ptr_type).contents

    def set_region(self, start, count, data):
        array_start = self.attribute_count * start
        array_end = self.attribute_count * count + array_start
        self.data[array_start:array_end] = data
        byte_start = self.attribute_stride * start
        byte_end = byte_start + self.attribute_stride * count
        if byte_start < self._dirty_min:
            self._dirty_min = byte_start
        if byte_end > self._dirty_max:
            self._dirty_max = byte_end
        self._dirty = True

    def resize(self, size):
        number = size // ctypes.sizeof(self.attribute_ctype)
        data = (self.attribute_ctype * number)()
        ctypes.memmove(data, self.data, min(size, self.size))
        self.data = data
        self.data_ptr = ctypes.addressof(data)
        self.size = size
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glBufferData(GL_ARRAY_BUFFER, self.size, self.data, self.usage)
        self._dirty_min = sys.maxsize
        self._dirty_max = 0
        self._dirty = False
        self.get_region.cache_clear()

    def invalidate(self):
        super().invalidate()
        self._dirty = True

    def invalidate_region(self, start, count):
        byte_start = self.attribute_stride * start
        byte_end = byte_start + self.attribute_stride * count
        if byte_start < self._dirty_min:
            self._dirty_min = byte_start
        if byte_end > self._dirty_max:
            self._dirty_max = byte_end
        self._dirty = True