from collections import Counter, defaultdict, deque
import networkx as nx
from networkx.utils import groups, not_implemented_for, py_random_state
def _fast_label_count(G, comms, node, weight=None):
    """Computes the frequency of labels in the neighborhood of a node.

    Returns a dictionary keyed by label to the frequency of that label.
    """
    if weight is None:
        if not G.is_multigraph():
            label_freqs = Counter(map(comms.get, nx.all_neighbors(G, node)))
        else:
            label_freqs = defaultdict(int)
            for nbr in G[node]:
                label_freqs[comms[nbr]] += len(G[node][nbr])
            if G.is_directed():
                for nbr in G.pred[node]:
                    label_freqs[comms[nbr]] += len(G.pred[node][nbr])
    else:
        label_freqs = defaultdict(float)
        for _, nbr, w in G.edges(node, data=weight, default=1):
            label_freqs[comms[nbr]] += w
        if G.is_directed():
            for nbr, _, w in G.in_edges(node, data=weight, default=1):
                label_freqs[comms[nbr]] += w
    return label_freqs