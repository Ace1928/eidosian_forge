from collections.abc import Mapping, Set
import networkx as nx
class MultiEdgeView(OutMultiEdgeView):
    """A EdgeView class for edges of a MultiGraph"""
    __slots__ = ()
    dataview = MultiEdgeDataView

    def __len__(self):
        return sum((1 for e in self))

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, kd in nbrs.items():
                if nbr not in seen:
                    for k, dd in kd.items():
                        yield (n, nbr, k)
            seen[n] = 1
        del seen