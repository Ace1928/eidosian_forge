from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
def _node_product(G, H):
    for u, v in product(G, H):
        yield ((u, v), _dict_product(G.nodes[u], H.nodes[v]))