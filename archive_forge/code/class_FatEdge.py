import networkx as nx
from collections import deque
class FatEdge(Edge):
    """
    An Edge that knows its place among the edges incident to each
    of its vertices.  Initialize with two pairs (v,n) and (w,m) meaning
    that this edge joins v to w and has index n at v and m at w.
    The parity of the optional integer argument twists determines
    whether the edge is twisted or not.
    """

    def __new__(cls, x, y, twists=0):
        return tuple.__new__(cls, (x[0], y[0]))

    def __init__(self, x, y, twists=0):
        self.slots = [x[1], y[1]]
        self.twisted = bool(twists % 2)

    def __repr__(self):
        return '%s[%d] -%s- %s[%d]' % (self[0], self.slots[0], 'x' if self.twisted else '-', self[1], self.slots[1])

    def __hash__(self):
        return id(self)

    def slot(self, vertex):
        try:
            return self.slots[self.index(vertex)]
        except ValueError:
            raise ValueError('Vertex is not an end of this edge.')

    def set_slot(self, vertex, n):
        try:
            self.slots[self.index(vertex)] = n
        except ValueError:
            raise ValueError('Vertex is not an end of this edge.')