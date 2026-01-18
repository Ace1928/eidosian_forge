import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def _edges_of_same_color(self, sgn1, sgn2):
    """
        Returns all edges in :attr:`graph` that have the same colour as the
        edge between sgn1 and sgn2 in :attr:`subgraph`.
        """
    if (sgn1, sgn2) in self._sge_colors:
        sge_color = self._sge_colors[sgn1, sgn2]
    else:
        sge_color = self._sge_colors[sgn2, sgn1]
    if sge_color in self._edge_compatibility:
        ge_color = self._edge_compatibility[sge_color]
        g_edges = self._ge_partitions[ge_color]
    else:
        g_edges = []
    return g_edges