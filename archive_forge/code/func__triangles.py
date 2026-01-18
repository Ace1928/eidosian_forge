from collections import defaultdict
from functools import partial
from itertools import combinations
import networkx as nx
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for
def _triangles(G, e):
    """Return list of all triangles containing edge e"""
    u, v = e
    if u not in G:
        raise nx.NetworkXError(f'Vertex {u} not in graph')
    if v not in G[u]:
        raise nx.NetworkXError(f'Edge ({u}, {v}) not in graph')
    triangle_list = []
    for x in G[u]:
        if x in G[v]:
            triangle_list.append((u, v, x))
    return triangle_list