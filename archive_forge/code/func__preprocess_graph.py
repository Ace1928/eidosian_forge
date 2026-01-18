from functools import partial
import networkx as nx
from networkx.utils import (
def _preprocess_graph(G, weight):
    """Compute edge weights and eliminate zero-weight edges."""
    if G.is_directed():
        H = nx.MultiGraph()
        H.add_nodes_from(G)
        H.add_weighted_edges_from(((u, v, e.get(weight, 1.0)) for u, v, e in G.edges(data=True) if u != v), weight=weight)
        G = H
    if not G.is_multigraph():
        edges = ((u, v, abs(e.get(weight, 1.0))) for u, v, e in G.edges(data=True) if u != v)
    else:
        edges = ((u, v, sum((abs(e.get(weight, 1.0)) for e in G[u][v].values()))) for u, v in G.edges() if u != v)
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_weighted_edges_from(((u, v, e) for u, v, e in edges if e != 0))
    return H