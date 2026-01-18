from collections.abc import Mapping, Set
import networkx as nx
class OutMultiEdgeView(OutEdgeView):
    """A EdgeView class for outward edges of a MultiDiGraph"""
    __slots__ = ()
    dataview = OutMultiEdgeDataView

    def __len__(self):
        return sum((len(kdict) for n, nbrs in self._nodes_nbrs() for nbr, kdict in nbrs.items()))

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr, kdict in nbrs.items():
                for key in kdict:
                    yield (n, nbr, key)

    def __contains__(self, e):
        N = len(e)
        if N == 3:
            u, v, k = e
        elif N == 2:
            u, v = e
            k = 0
        else:
            raise ValueError('MultiEdge must have length 2 or 3')
        try:
            return k in self._adjdict[u][v]
        except KeyError:
            return False

    def __getitem__(self, e):
        if isinstance(e, slice):
            raise nx.NetworkXError(f'{type(self).__name__} does not support slicing, try list(G.edges)[{e.start}:{e.stop}:{e.step}]')
        u, v, k = e
        return self._adjdict[u][v][k]

    def __call__(self, nbunch=None, data=False, *, default=None, keys=False):
        if nbunch is None and data is False and (keys is True):
            return self
        return self.dataview(self, nbunch, data, default=default, keys=keys)

    def data(self, data=True, default=None, nbunch=None, keys=False):
        if nbunch is None and data is False and (keys is True):
            return self
        return self.dataview(self, nbunch, data, default=default, keys=keys)