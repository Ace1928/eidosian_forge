from collections.abc import Mapping, Set
import networkx as nx
class OutMultiEdgeDataView(OutEdgeDataView):
    """An EdgeDataView for outward edges of MultiDiGraph; See EdgeDataView"""
    __slots__ = ('keys',)

    def __getstate__(self):
        return {'viewer': self._viewer, 'nbunch': self._nbunch, 'keys': self.keys, 'data': self._data, 'default': self._default}

    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, viewer, nbunch=None, data=False, *, default=None, keys=False):
        self._viewer = viewer
        adjdict = self._adjdict = viewer._adjdict
        self.keys = keys
        if nbunch is None:
            self._nodes_nbrs = adjdict.items
        else:
            nbunch = dict.fromkeys(viewer._graph.nbunch_iter(nbunch))
            self._nodes_nbrs = lambda: [(n, adjdict[n]) for n in nbunch]
        self._nbunch = nbunch
        self._data = data
        self._default = default
        if data is True:
            if keys is True:
                self._report = lambda n, nbr, k, dd: (n, nbr, k, dd)
            else:
                self._report = lambda n, nbr, k, dd: (n, nbr, dd)
        elif data is False:
            if keys is True:
                self._report = lambda n, nbr, k, dd: (n, nbr, k)
            else:
                self._report = lambda n, nbr, k, dd: (n, nbr)
        elif keys is True:
            self._report = lambda n, nbr, k, dd: (n, nbr, k, dd[data]) if data in dd else (n, nbr, k, default)
        else:
            self._report = lambda n, nbr, k, dd: (n, nbr, dd[data]) if data in dd else (n, nbr, default)

    def __len__(self):
        return sum((1 for e in self))

    def __iter__(self):
        return (self._report(n, nbr, k, dd) for n, nbrs in self._nodes_nbrs() for nbr, kd in nbrs.items() for k, dd in kd.items())

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch:
            return False
        try:
            kdict = self._adjdict[u][v]
        except KeyError:
            return False
        if self.keys is True:
            k = e[2]
            try:
                dd = kdict[k]
            except KeyError:
                return False
            return e == self._report(u, v, k, dd)
        return any((e == self._report(u, v, k, dd) for k, dd in kdict.items()))