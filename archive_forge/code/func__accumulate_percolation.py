import networkx as nx
from networkx.algorithms.centrality.betweenness import (
from networkx.algorithms.centrality.betweenness import (
def _accumulate_percolation(percolation, S, P, sigma, s, states, p_sigma_x_t):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            pw_s_w = states[s] / (p_sigma_x_t - states[w])
            percolation[w] += delta[w] * pw_s_w
    return percolation