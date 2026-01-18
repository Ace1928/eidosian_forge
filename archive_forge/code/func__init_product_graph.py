from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
def _init_product_graph(G, H):
    if G.is_directed() != H.is_directed():
        msg = 'G and H must be both directed or both undirected'
        raise nx.NetworkXError(msg)
    if G.is_multigraph() or H.is_multigraph():
        GH = nx.MultiGraph()
    else:
        GH = nx.Graph()
    if G.is_directed():
        GH = GH.to_directed()
    return GH