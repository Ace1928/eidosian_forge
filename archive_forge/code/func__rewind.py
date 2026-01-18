from collections import deque
import networkx as nx
def _rewind(self):
    self._it = iter(self._edges.items())
    self._curr = next(self._it)