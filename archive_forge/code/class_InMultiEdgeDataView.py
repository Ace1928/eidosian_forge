from collections.abc import Mapping, Set
import networkx as nx
class InMultiEdgeDataView(OutMultiEdgeDataView):
    """An EdgeDataView for inward edges of MultiDiGraph; See EdgeDataView"""
    __slots__ = ()

    def __iter__(self):
        return (self._report(nbr, n, k, dd) for n, nbrs in self._nodes_nbrs() for nbr, kd in nbrs.items() for k, dd in kd.items())

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and v not in self._nbunch:
            return False
        try:
            kdict = self._adjdict[v][u]
        except KeyError:
            return False
        if self.keys is True:
            k = e[2]
            dd = kdict[k]
            return e == self._report(u, v, k, dd)
        return any((e == self._report(u, v, k, dd) for k, dd in kdict.items()))