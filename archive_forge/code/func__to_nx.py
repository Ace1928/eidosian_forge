import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _to_nx(edges, n_nodes, root=None, roots=None):
    """
    Converts the (edges, n_nodes) input to a :class:`networkx.Graph`.
    The (edges, n_nodes) input is a list of even length, where each pair
    of consecutive integers represents an edge, and an integer `n_nodes`.
    Integers in the list are elements of `range(n_nodes)`.

    Parameters
    ----------
    edges : list of ints
        The flattened list of edges of the graph.
    n_nodes : int
        The number of nodes of the graph.
    root: int (default=None)
        If not None, the "root" attribute of the graph will be set to this value.
    roots: collection of ints (default=None)
        If not None, he "roots" attribute of the graph will be set to this value.

    Returns
    -------
    :class:`networkx.Graph`
        The graph with `n_nodes` nodes and edges given by `edges`.
    """
    G = nx.empty_graph(n_nodes)
    G.add_edges_from(edges)
    if root is not None:
        G.graph['root'] = root
    if roots is not None:
        G.graph['roots'] = roots
    return G