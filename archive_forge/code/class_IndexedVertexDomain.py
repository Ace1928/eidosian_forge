import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
class IndexedVertexDomain(VertexDomain):
    """Management of a set of indexed vertex lists.

    Construction of an indexed vertex domain is usually done with the
    :py:func:`create_domain` function.
    """
    _initial_index_count = 16

    def __init__(self, program, attribute_meta, index_gl_type=GL_UNSIGNED_INT):
        super(IndexedVertexDomain, self).__init__(program, attribute_meta)
        self.index_allocator = allocation.Allocator(self._initial_index_count)
        self.index_gl_type = index_gl_type
        self.index_c_type = shader._c_types[index_gl_type]
        self.index_element_size = ctypes.sizeof(self.index_c_type)
        self.index_buffer = BufferObject(self.index_allocator.capacity * self.index_element_size)
        self.vao.bind()
        self.index_buffer.bind_to_index_buffer()
        self.vao.unbind()
        self._vertexlist_class = type('IndexedVertexList', (IndexedVertexList,), self._property_dict)

    def safe_index_alloc(self, count):
        """Allocate indices, resizing the buffers if necessary."""
        try:
            return self.index_allocator.alloc(count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            self.index_buffer.resize(capacity * self.index_element_size)
            self.index_allocator.set_capacity(capacity)
            return self.index_allocator.alloc(count)

    def safe_index_realloc(self, start, count, new_count):
        """Reallocate indices, resizing the buffers if necessary."""
        try:
            return self.index_allocator.realloc(start, count, new_count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            self.index_buffer.resize(capacity * self.index_element_size)
            self.index_allocator.set_capacity(capacity)
            return self.index_allocator.realloc(start, count, new_count)

    def create(self, count, index_count):
        """Create an :py:class:`IndexedVertexList` in this domain.

        :Parameters:
            `count` : int
                Number of vertices to create
            `index_count`
                Number of indices to create

        """
        start = self.safe_alloc(count)
        index_start = self.safe_index_alloc(index_count)
        return self._vertexlist_class(self, start, count, index_start, index_count)

    def get_index_region(self, start, count):
        """Get a data from a region of the index buffer.

        :Parameters:
            `start` : int
                Start of the region to map.
            `count` : int
                Number of indices to map.

        :rtype: Array of int
        """
        byte_start = self.index_element_size * start
        byte_count = self.index_element_size * count
        ptr_type = ctypes.POINTER(self.index_c_type * count)
        map_ptr = self.index_buffer.map_range(byte_start, byte_count, ptr_type)
        data = map_ptr[:]
        self.index_buffer.unmap()
        return data

    def set_index_region(self, start, count, data):
        byte_start = self.index_element_size * start
        byte_count = self.index_element_size * count
        ptr_type = ctypes.POINTER(self.index_c_type * count)
        map_ptr = self.index_buffer.map_range(byte_start, byte_count, ptr_type)
        map_ptr[:] = data
        self.index_buffer.unmap()

    def draw(self, mode):
        """Draw all vertices in the domain.

        All vertices in the domain are drawn at once. This is the
        most efficient way to render primitives.

        :Parameters:
            `mode` : int
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.sub_data()
        starts, sizes = self.index_allocator.get_allocated_regions()
        primcount = len(starts)
        if primcount == 0:
            pass
        elif primcount == 1:
            glDrawElements(mode, sizes[0], self.index_gl_type, self.index_buffer.ptr + starts[0] * self.index_element_size)
        else:
            starts = [s * self.index_element_size + self.index_buffer.ptr for s in starts]
            starts = (ctypes.POINTER(GLvoid) * primcount)(*(GLintptr * primcount)(*starts))
            sizes = (GLsizei * primcount)(*sizes)
            glMultiDrawElements(mode, sizes, self.index_gl_type, starts, primcount)

    def draw_subset(self, mode, vertex_list):
        """Draw a specific IndexedVertexList in the domain.

        The `vertex_list` parameter specifies a :py:class:`IndexedVertexList`
        to draw. Only primitives in that list will be drawn.

        :Parameters:
            `mode` : int
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.
            `vertex_list` : `IndexedVertexList`
                Vertex list to draw.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.sub_data()
        glDrawElements(mode, vertex_list.index_count, self.index_gl_type, self.index_buffer.ptr + vertex_list.index_start * self.index_element_size)