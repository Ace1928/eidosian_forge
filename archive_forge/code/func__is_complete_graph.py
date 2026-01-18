import sys
import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.utils import arbitrary_element, not_implemented_for
def _is_complete_graph(G):
    """Returns True if G is a complete graph."""
    if nx.number_of_selfloops(G) > 0:
        raise nx.NetworkXError('Self loop found in _is_complete_graph()')
    n = G.number_of_nodes()
    if n < 2:
        return True
    e = G.number_of_edges()
    max_edges = n * (n - 1) / 2
    return e == max_edges