from collections.abc import Mapping, Set
import networkx as nx
class OutEdgeDataView:
    """EdgeDataView for outward edges of DiGraph; See EdgeDataView"""
    __slots__ = ('_viewer', '_nbunch', '_data', '_default', '_adjdict', '_nodes_nbrs', '_report')

    def __getstate__(self):
        return {'viewer': self._viewer, 'nbunch': self._nbunch, 'data': self._data, 'default': self._default}

    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, viewer, nbunch=None, data=False, *, default=None):
        self._viewer = viewer
        adjdict = self._adjdict = viewer._adjdict
        if nbunch is None:
            self._nodes_nbrs = adjdict.items
        else:
            nbunch = dict.fromkeys(viewer._graph.nbunch_iter(nbunch))
            self._nodes_nbrs = lambda: [(n, adjdict[n]) for n in nbunch]
        self._nbunch = nbunch
        self._data = data
        self._default = default
        if data is True:
            self._report = lambda n, nbr, dd: (n, nbr, dd)
        elif data is False:
            self._report = lambda n, nbr, dd: (n, nbr)
        else:
            self._report = lambda n, nbr, dd: (n, nbr, dd[data]) if data in dd else (n, nbr, default)

    def __len__(self):
        return sum((len(nbrs) for n, nbrs in self._nodes_nbrs()))

    def __iter__(self):
        return (self._report(n, nbr, dd) for n, nbrs in self._nodes_nbrs() for nbr, dd in nbrs.items())

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch:
            return False
        try:
            ddict = self._adjdict[u][v]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self)})'