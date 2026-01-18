import networkx as nx
from collections import deque
class DirectedEdge(BaseEdge):
    """
    An Edge with a tail and a head.  The two vertices can be accessed as
    E.tail and E.head
    """

    def __repr__(self):
        return '%s --> %s' % self

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    def head(self):
        return self[1]

    @property
    def tail(self):
        return self[0]