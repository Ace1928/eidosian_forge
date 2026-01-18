from collections.abc import Mapping, Set
import networkx as nx
class NodeDataView(Set):
    """A DataView class for nodes of a NetworkX Graph

    The main use for this class is to iterate through node-data pairs.
    The data can be the entire data-dictionary for each node, or it
    can be a specific attribute (with default) for each node.
    Set operations are enabled with NodeDataView, but don't work in
    cases where the data is not hashable. Use with caution.
    Typically, set operations on nodes use NodeView, not NodeDataView.
    That is, they use `G.nodes` instead of `G.nodes(data='foo')`.

    Parameters
    ==========
    graph : NetworkX graph-like class
    data : bool or string (default=False)
    default : object (default=None)
    """
    __slots__ = ('_nodes', '_data', '_default')

    def __getstate__(self):
        return {'_nodes': self._nodes, '_data': self._data, '_default': self._default}

    def __setstate__(self, state):
        self._nodes = state['_nodes']
        self._data = state['_data']
        self._default = state['_default']

    def __init__(self, nodedict, data=False, default=None):
        self._nodes = nodedict
        self._data = data
        self._default = default

    @classmethod
    def _from_iterable(cls, it):
        try:
            return set(it)
        except TypeError as err:
            if 'unhashable' in str(err):
                msg = ' : Could be b/c data=True or your values are unhashable'
                raise TypeError(str(err) + msg) from err
            raise

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        data = self._data
        if data is False:
            return iter(self._nodes)
        if data is True:
            return iter(self._nodes.items())
        return ((n, dd[data] if data in dd else self._default) for n, dd in self._nodes.items())

    def __contains__(self, n):
        try:
            node_in = n in self._nodes
        except TypeError:
            n, d = n
            return n in self._nodes and self[n] == d
        if node_in is True:
            return node_in
        try:
            n, d = n
        except (TypeError, ValueError):
            return False
        return n in self._nodes and self[n] == d

    def __getitem__(self, n):
        if isinstance(n, slice):
            raise nx.NetworkXError(f'{type(self).__name__} does not support slicing, try list(G.nodes.data())[{n.start}:{n.stop}:{n.step}]')
        ddict = self._nodes[n]
        data = self._data
        if data is False or data is True:
            return ddict
        return ddict[data] if data in ddict else self._default

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        name = self.__class__.__name__
        if self._data is False:
            return f'{name}({tuple(self)})'
        if self._data is True:
            return f'{name}({dict(self)})'
        return f'{name}({dict(self)}, data={self._data!r})'