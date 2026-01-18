import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
class IndexedVertexList(VertexList):
    """A list of vertices within an :py:class:`IndexedVertexDomain` that are
    indexed. Use :py:meth:`IndexedVertexDomain.create` to construct this list.
    """
    _indices_cache = None
    _indices_cache_version = None

    def __init__(self, domain, start, count, index_start, index_count):
        super().__init__(domain, start, count)
        self.index_start = index_start
        self.index_count = index_count

    def resize(self, count, index_count):
        """Resize this group.

        :Parameters:
            `count` : int
                New number of vertices in the list.
            `index_count` : int
                New number of indices in the list.

        """
        old_start = self.start
        super().resize(count)
        if old_start != self.start:
            diff = self.start - old_start
            self.indices[:] = [i + diff for i in self.indices]
        new_start = self.domain.safe_index_realloc(self.index_start, self.index_count, index_count)
        if new_start != self.index_start:
            old = self.domain.get_index_region(self.index_start, self.index_count)
            new = self.domain.get_index_region(self.index_start, self.index_count)
            new.array[:] = old.array[:]
            new.invalidate()
        self.index_start = new_start
        self.index_count = index_count
        self._indices_cache_version = None

    def delete(self):
        """Delete this group."""
        super().delete()
        self.domain.index_allocator.dealloc(self.index_start, self.index_count)

    def migrate(self, domain):
        """Move this group from its current indexed domain and add to the
        specified one.  Attributes on domains must match.  (In practice, used 
        to change parent state of some vertices).

        :Parameters:
            `domain` : `IndexedVertexDomain`
                Indexed domain to migrate this vertex list to.

        """
        old_start = self.start
        old_domain = self.domain
        super().migrate(domain)
        if old_start != self.start:
            diff = self.start - old_start
            old_indices = old_domain.get_index_region(self.index_start, self.index_count)
            old_domain.set_index_region(self.index_start, self.index_count, [i + diff for i in old_indices])
        old_array = old_domain.get_index_region(self.index_start, self.index_count)
        old_domain.index_allocator.dealloc(self.index_start, self.index_count)
        new_start = self.domain.safe_index_alloc(self.index_count)
        self.domain.set_index_region(new_start, self.index_count, old_array)
        self.index_start = new_start
        self._indices_cache_version = None

    @property
    def indices(self):
        """Array of index data."""
        if self._indices_cache_version != self.domain.version:
            domain = self.domain
            self._indices_cache = domain.get_index_region(self.index_start, self.index_count)
            self._indices_cache_version = domain.version
        return self._indices_cache

    @indices.setter
    def indices(self, data):
        self.domain.set_index_region(self.index_start, self.index_count, data)