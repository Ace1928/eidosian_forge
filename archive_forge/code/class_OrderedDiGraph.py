import collections
import io
import os
import networkx as nx
from networkx.drawing import nx_pydot
class OrderedDiGraph(DiGraph):
    """A directed graph subclass with useful utility functions.

    This derivative retains node, edge, insertion and iteration
    ordering (so that the iteration order matches the insertion
    order).
    """
    node_dict_factory = collections.OrderedDict
    adjlist_outer_dict_factory = collections.OrderedDict
    adjlist_inner_dict_factory = collections.OrderedDict
    edge_attr_dict_factory = collections.OrderedDict

    def fresh_copy(self):
        """Return a fresh copy graph with the same data structure.

        A fresh copy has no nodes, edges or graph attributes. It is
        the same data structure as the current graph. This method is
        typically used to create an empty version of the graph.
        """
        return OrderedDiGraph()