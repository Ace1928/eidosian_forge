from itertools import chain
from math import log
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for
def _build_flow_dict(G, R, capacity, weight):
    """Build a flow dictionary from a residual network."""
    inf = float('inf')
    flow_dict = {}
    if G.is_multigraph():
        for u in G:
            flow_dict[u] = {}
            for v, es in G[u].items():
                flow_dict[u][v] = {k: 0 if u != v or e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0 else e[capacity] for k, e in es.items()}
            for v, es in R[u].items():
                if v in flow_dict[u]:
                    flow_dict[u][v].update(((k[0], e['flow']) for k, e in es.items() if e['flow'] > 0))
    else:
        for u in G:
            flow_dict[u] = {v: 0 if u != v or e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0 else e[capacity] for v, e in G[u].items()}
            flow_dict[u].update(((v, e['flow']) for v, es in R[u].items() for e in es.values() if e['flow'] > 0))
    return flow_dict