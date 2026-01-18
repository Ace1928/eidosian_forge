import time
from . import debug, errors, osutils, revision, trace
class HeadsCache:
    """A cache of results for graph heads calls."""

    def __init__(self, graph):
        self.graph = graph
        self._heads = {}

    def heads(self, keys):
        """Return the heads of keys.

        This matches the API of Graph.heads(), specifically the return value is
        a set which can be mutated, and ordering of the input is not preserved
        in the output.

        :see also: Graph.heads.
        :param keys: The keys to calculate heads for.
        :return: A set containing the heads, which may be mutated without
            affecting future lookups.
        """
        keys = frozenset(keys)
        try:
            return set(self._heads[keys])
        except KeyError:
            heads = self.graph.heads(keys)
            self._heads[keys] = heads
            return set(heads)