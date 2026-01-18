from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
def _directed_edges_cross_edges(G, H):
    if not G.is_multigraph() and (not H.is_multigraph()):
        for u, v, c in G.edges(data=True):
            for x, y, d in H.edges(data=True):
                yield ((u, x), (v, y), _dict_product(c, d))
    if not G.is_multigraph() and H.is_multigraph():
        for u, v, c in G.edges(data=True):
            for x, y, k, d in H.edges(data=True, keys=True):
                yield ((u, x), (v, y), k, _dict_product(c, d))
    if G.is_multigraph() and (not H.is_multigraph()):
        for u, v, k, c in G.edges(data=True, keys=True):
            for x, y, d in H.edges(data=True):
                yield ((u, x), (v, y), k, _dict_product(c, d))
    if G.is_multigraph() and H.is_multigraph():
        for u, v, j, c in G.edges(data=True, keys=True):
            for x, y, k, d in H.edges(data=True, keys=True):
                yield ((u, x), (v, y), (j, k), _dict_product(c, d))