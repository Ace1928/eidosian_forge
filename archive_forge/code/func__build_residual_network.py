from itertools import chain
from math import log
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for
@not_implemented_for('undirected')
def _build_residual_network(G, demand, capacity, weight):
    """Build a residual network and initialize a zero flow."""
    if sum((G.nodes[u].get(demand, 0) for u in G)) != 0:
        raise nx.NetworkXUnfeasible('Sum of the demands should be 0.')
    R = nx.MultiDiGraph()
    R.add_nodes_from(((u, {'excess': -G.nodes[u].get(demand, 0), 'potential': 0}) for u in G))
    inf = float('inf')
    for u, v, e in nx.selfloop_edges(G, data=True):
        if e.get(weight, 0) < 0 and e.get(capacity, inf) == inf:
            raise nx.NetworkXUnbounded('Negative cost cycle of infinite capacity found. Min cost flow may be unbounded below.')
    if G.is_multigraph():
        edge_list = [(u, v, k, e) for u, v, k, e in G.edges(data=True, keys=True) if u != v and e.get(capacity, inf) > 0]
    else:
        edge_list = [(u, v, 0, e) for u, v, e in G.edges(data=True) if u != v and e.get(capacity, inf) > 0]
    inf = max(sum((abs(R.nodes[u]['excess']) for u in R)), 2 * sum((e[capacity] for u, v, k, e in edge_list if capacity in e and e[capacity] != inf))) or 1
    for u, v, k, e in edge_list:
        r = min(e.get(capacity, inf), inf)
        w = e.get(weight, 0)
        R.add_edge(u, v, key=(k, True), capacity=r, weight=w, flow=0)
        R.add_edge(v, u, key=(k, False), capacity=0, weight=-w, flow=0)
    R.graph['inf'] = inf
    _detect_unboundedness(R)
    return R