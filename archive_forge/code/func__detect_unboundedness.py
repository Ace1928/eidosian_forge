from itertools import chain
from math import log
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for
def _detect_unboundedness(R):
    """Detect infinite-capacity negative cycles."""
    G = nx.DiGraph()
    G.add_nodes_from(R)
    inf = R.graph['inf']
    f_inf = float('inf')
    for u in R:
        for v, e in R[u].items():
            w = f_inf
            for k, e in e.items():
                if e['capacity'] == inf:
                    w = min(w, e['weight'])
            if w != f_inf:
                G.add_edge(u, v, weight=w)
    if nx.negative_edge_cycle(G):
        raise nx.NetworkXUnbounded('Negative cost cycle of infinite capacity found. Min cost flow may be unbounded below.')