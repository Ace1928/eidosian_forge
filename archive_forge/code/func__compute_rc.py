from itertools import accumulate
import networkx as nx
from networkx.utils import not_implemented_for
def _compute_rc(G):
    """Returns the rich-club coefficient for each degree in the graph
    `G`.

    `G` is an undirected graph without multiedges.

    Returns a dictionary mapping degree to rich-club coefficient for
    that degree.

    """
    deghist = nx.degree_histogram(G)
    total = sum(deghist)
    nks = (total - cs for cs in accumulate(deghist) if total - cs > 1)
    edge_degrees = sorted((sorted(map(G.degree, e)) for e in G.edges()), reverse=True)
    ek = G.number_of_edges()
    k1, k2 = edge_degrees.pop()
    rc = {}
    for d, nk in enumerate(nks):
        while k1 <= d:
            if len(edge_degrees) == 0:
                ek = 0
                break
            k1, k2 = edge_degrees.pop()
            ek -= 1
        rc[d] = 2 * ek / (nk * (nk - 1))
    return rc