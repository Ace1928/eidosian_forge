from collections.abc import Mapping, Set
import networkx as nx
class OutEdgeView(Set, Mapping):
    """A EdgeView class for outward edges of a DiGraph"""
    __slots__ = ('_adjdict', '_graph', '_nodes_nbrs')

    def __getstate__(self):
        return {'_graph': self._graph, '_adjdict': self._adjdict}

    def __setstate__(self, state):
        self._graph = state['_graph']
        self._adjdict = state['_adjdict']
        self._nodes_nbrs = self._adjdict.items

    @classmethod
    def _from_iterable(cls, it):
        return set(it)
    dataview = OutEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._succ if hasattr(G, 'succ') else G._adj
        self._nodes_nbrs = self._adjdict.items

    def __len__(self):
        return sum((len(nbrs) for n, nbrs in self._nodes_nbrs()))

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr in nbrs:
                yield (n, nbr)

    def __contains__(self, e):
        try:
            u, v = e
            return v in self._adjdict[u]
        except KeyError:
            return False

    def __getitem__(self, e):
        if isinstance(e, slice):
            raise nx.NetworkXError(f'{type(self).__name__} does not support slicing, try list(G.edges)[{e.start}:{e.stop}:{e.step}]')
        u, v = e
        return self._adjdict[u][v]

    def __call__(self, nbunch=None, data=False, *, default=None):
        if nbunch is None and data is False:
            return self
        return self.dataview(self, nbunch, data, default=default)

    def data(self, data=True, default=None, nbunch=None):
        """
        Return a read-only view of edge data.

        Parameters
        ----------
        data : bool or edge attribute key
            If ``data=True``, then the data view maps each edge to a dictionary
            containing all of its attributes. If `data` is a key in the edge
            dictionary, then the data view maps each edge to its value for
            the keyed attribute. In this case, if the edge doesn't have the
            attribute, the `default` value is returned.
        default : object, default=None
            The value used when an edge does not have a specific attribute
        nbunch : container of nodes, optional (default=None)
            Allows restriction to edges only involving certain nodes. All edges
            are considered by default.

        Returns
        -------
        dataview
            Returns an `EdgeDataView` for undirected Graphs, `OutEdgeDataView`
            for DiGraphs, `MultiEdgeDataView` for MultiGraphs and
            `OutMultiEdgeDataView` for MultiDiGraphs.

        Notes
        -----
        If ``data=False``, returns an `EdgeView` without any edge data.

        See Also
        --------
        EdgeDataView
        OutEdgeDataView
        MultiEdgeDataView
        OutMultiEdgeDataView

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_edges_from([
        ...     (0, 1, {"dist": 3, "capacity": 20}),
        ...     (1, 2, {"dist": 4}),
        ...     (2, 0, {"dist": 5})
        ... ])

        Accessing edge data with ``data=True`` (the default) returns an
        edge data view object listing each edge with all of its attributes:

        >>> G.edges.data()
        EdgeDataView([(0, 1, {'dist': 3, 'capacity': 20}), (0, 2, {'dist': 5}), (1, 2, {'dist': 4})])

        If `data` represents a key in the edge attribute dict, a dataview listing
        each edge with its value for that specific key is returned:

        >>> G.edges.data("dist")
        EdgeDataView([(0, 1, 3), (0, 2, 5), (1, 2, 4)])

        `nbunch` can be used to limit the edges:

        >>> G.edges.data("dist", nbunch=[0])
        EdgeDataView([(0, 1, 3), (0, 2, 5)])

        If a specific key is not found in an edge attribute dict, the value
        specified by `default` is used:

        >>> G.edges.data("capacity")
        EdgeDataView([(0, 1, 20), (0, 2, None), (1, 2, None)])

        Note that there is no check that the `data` key is present in any of
        the edge attribute dictionaries:

        >>> G.edges.data("speed")
        EdgeDataView([(0, 1, None), (0, 2, None), (1, 2, None)])
        """
        if nbunch is None and data is False:
            return self
        return self.dataview(self, nbunch, data, default=default)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self)})'