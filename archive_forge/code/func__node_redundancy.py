from itertools import combinations
import networkx as nx
from networkx import NetworkXError
def _node_redundancy(G, v):
    """Returns the redundancy of the node `v` in the bipartite graph `G`.

    If `G` is a graph with `n` nodes, the redundancy of a node is the ratio
    of the "overlap" of `v` to the maximum possible overlap of `v`
    according to its degree. The overlap of `v` is the number of pairs of
    neighbors that have mutual neighbors themselves, other than `v`.

    `v` must have at least two neighbors in `G`.

    """
    n = len(G[v])
    overlap = sum((1 for u, w in combinations(G[v], 2) if (set(G[u]) & set(G[w])) - {v}))
    return 2 * overlap / (n * (n - 1))