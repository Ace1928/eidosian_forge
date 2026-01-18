from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _kou_steiner_tree(G, terminal_nodes, weight):
    M = metric_closure(G, weight=weight)
    H = M.subgraph(terminal_nodes)
    mst_edges = nx.minimum_spanning_edges(H, weight='distance', data=True)
    mst_all_edges = chain.from_iterable((pairwise(d['path']) for u, v, d in mst_edges))
    if G.is_multigraph():
        mst_all_edges = ((u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for u, v in mst_all_edges)
    G_S = G.edge_subgraph(mst_all_edges)
    T_S = nx.minimum_spanning_edges(G_S, weight='weight', data=False)
    T_H = G.edge_subgraph(T_S).copy()
    _remove_nonterminal_leaves(T_H, terminal_nodes)
    return T_H.edges()