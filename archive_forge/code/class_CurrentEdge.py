from collections import deque
import networkx as nx
class CurrentEdge:
    """Mechanism for iterating over out-edges incident to a node in a circular
    manner. StopIteration exception is raised when wraparound occurs.
    """
    __slots__ = ('_edges', '_it', '_curr')

    def __init__(self, edges):
        self._edges = edges
        if self._edges:
            self._rewind()

    def get(self):
        return self._curr

    def move_to_next(self):
        try:
            self._curr = next(self._it)
        except StopIteration:
            self._rewind()
            raise

    def _rewind(self):
        self._it = iter(self._edges.items())
        self._curr = next(self._it)