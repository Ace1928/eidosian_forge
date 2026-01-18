import networkx as nx
from networkx.algorithms.centrality.betweenness import (
from networkx.algorithms.centrality.betweenness import (
from networkx.algorithms.centrality.betweenness import (
def _accumulate_edges_subset(betweenness, S, P, sigma, s, targets):
    """edge_betweenness_centrality_subset helper."""
    delta = dict.fromkeys(S, 0)
    target_set = set(targets)
    while S:
        w = S.pop()
        for v in P[w]:
            if w in target_set:
                c = sigma[v] / sigma[w] * (1.0 + delta[w])
            else:
                c = delta[w] / len(P[w])
            if (v, w) not in betweenness:
                betweenness[w, v] += c
            else:
                betweenness[v, w] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness