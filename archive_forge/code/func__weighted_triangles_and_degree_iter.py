from collections import Counter
from itertools import chain, combinations
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
def _weighted_triangles_and_degree_iter(G, nodes=None, weight='weight'):
    """Return an iterator of (node, degree, weighted_triangles).

    Used for weighted clustering.
    Note: this returns the geometric average weight of edges in the triangle.
    Also, each triangle is counted twice (each direction).
    So you may want to divide by 2.

    """
    import numpy as np
    if weight is None or G.number_of_edges() == 0:
        max_weight = 1
    else:
        max_weight = max((d.get(weight, 1) for u, v, d in G.edges(data=True)))
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight
    for i, nbrs in nodes_nbrs:
        inbrs = set(nbrs) - {i}
        weighted_triangles = 0
        seen = set()
        for j in inbrs:
            seen.add(j)
            jnbrs = set(G[j]) - seen
            wij = wt(i, j)
            weighted_triangles += sum(np.cbrt([wij * wt(j, k) * wt(k, i) for k in inbrs & jnbrs]))
        yield (i, len(inbrs), 2 * weighted_triangles)