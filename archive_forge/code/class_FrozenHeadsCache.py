import time
from . import debug, errors, osutils, revision, trace
class FrozenHeadsCache:
    """Cache heads() calls, assuming the caller won't modify them."""

    def __init__(self, graph):
        self.graph = graph
        self._heads = {}

    def heads(self, keys):
        """Return the heads of keys.

        Similar to Graph.heads(). The main difference is that the return value
        is a frozen set which cannot be mutated.

        :see also: Graph.heads.
        :param keys: The keys to calculate heads for.
        :return: A frozenset containing the heads.
        """
        keys = frozenset(keys)
        try:
            return self._heads[keys]
        except KeyError:
            heads = frozenset(self.graph.heads(keys))
            self._heads[keys] = heads
            return heads

    def cache(self, keys, heads):
        """Store a known value."""
        self._heads[frozenset(keys)] = frozenset(heads)