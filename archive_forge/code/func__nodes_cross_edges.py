from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
def _nodes_cross_edges(G, H):
    if H.is_multigraph():
        for x in G:
            for u, v, k, d in H.edges(data=True, keys=True):
                yield ((x, u), (x, v), k, d)
    else:
        for x in G:
            for u, v, d in H.edges(data=True):
                if G.is_multigraph():
                    yield ((x, u), (x, v), None, d)
                else:
                    yield ((x, u), (x, v), d)