import math
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _add_edge_to_spanner(H, residual_graph, u, v, weight):
    """Add the edge {u, v} to the spanner H and take weight from
    the residual graph.

    Parameters
    ----------
    H : NetworkX graph
        The spanner under construction.

    residual_graph : NetworkX graph
        The residual graph used by the Baswana-Sen algorithm. The weight
        for the edge is taken from this graph.

    u : node
        One endpoint of the edge.

    v : node
        The other endpoint of the edge.

    weight : object
        The edge attribute to use as distance.
    """
    H.add_edge(u, v)
    if weight:
        H[u][v][weight] = residual_graph[u][v]['weight'][0]