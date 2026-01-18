import networkx as nx
from networkx import NetworkXError
from networkx.utils import not_implemented_for
def _get_broadcast_centers(G, v, values, target):
    adj = sorted(G.neighbors(v), key=values.get, reverse=True)
    j = next((i for i, u in enumerate(adj, start=1) if values[u] + i == target))
    return set([v] + adj[:j])