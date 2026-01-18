import networkx as nx
from networkx import NetworkXError
from networkx.utils import not_implemented_for
def _get_max_broadcast_value(G, U, v, values):
    adj = sorted(set(G.neighbors(v)) & U, key=values.get, reverse=True)
    return max((values[u] + i for i, u in enumerate(adj, start=1)))