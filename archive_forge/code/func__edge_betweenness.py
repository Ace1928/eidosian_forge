from operator import itemgetter
import networkx as nx
def _edge_betweenness(G, source, nodes=None, cutoff=False):
    """Edge betweenness helper."""
    pred, length = nx.predecessor(G, source, cutoff=cutoff, return_seen=True)
    onodes = [n for n, d in sorted(length.items(), key=itemgetter(1))]
    between = {}
    for u, v in G.edges(nodes):
        between[u, v] = 1.0
        between[v, u] = 1.0
    while onodes:
        v = onodes.pop()
        if v in pred:
            num_paths = len(pred[v])
            for w in pred[v]:
                if w in pred:
                    num_paths = len(pred[w])
                    for x in pred[w]:
                        between[w, x] += between[v, w] / num_paths
                        between[x, w] += between[w, v] / num_paths
    return between