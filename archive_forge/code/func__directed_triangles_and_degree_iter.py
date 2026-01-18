from collections import Counter
from itertools import chain, combinations
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
def _directed_triangles_and_degree_iter(G, nodes=None):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_triangles).

    Used for directed clustering.
    Note that unlike `_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """
    nodes_nbrs = ((n, G._pred[n], G._succ[n]) for n in G.nbunch_iter(nodes))
    for i, preds, succs in nodes_nbrs:
        ipreds = set(preds) - {i}
        isuccs = set(succs) - {i}
        directed_triangles = 0
        for j in chain(ipreds, isuccs):
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._succ[j]) - {j}
            directed_triangles += sum((1 for k in chain(ipreds & jpreds, ipreds & jsuccs, isuccs & jpreds, isuccs & jsuccs)))
        dtotal = len(ipreds) + len(isuccs)
        dbidirectional = len(ipreds & isuccs)
        yield (i, dtotal, dbidirectional, directed_triangles)