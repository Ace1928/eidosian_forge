import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class VertexCacheBase:
    """Base class for a vertex cache for a simplicial complex."""

    def __init__(self):
        self.cache = collections.OrderedDict()
        self.nfev = 0
        self.index = -1

    def __iter__(self):
        for v in self.cache:
            yield self.cache[v]
        return

    def size(self):
        """Returns the size of the vertex cache."""
        return self.index + 1

    def print_out(self):
        headlen = len(f'Vertex cache of size: {len(self.cache)}:')
        print('=' * headlen)
        print(f'Vertex cache of size: {len(self.cache)}:')
        print('=' * headlen)
        for v in self.cache:
            self.cache[v].print_out()