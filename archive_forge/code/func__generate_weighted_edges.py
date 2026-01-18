import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
def _generate_weighted_edges(A):
    """Returns an iterable over (u, v, w) triples, where u and v are adjacent
    vertices and w is the weight of the edge joining u and v.

    `A` is a SciPy sparse array (in any format).

    """
    if A.format == 'csr':
        return _csr_gen_triples(A)
    if A.format == 'csc':
        return _csc_gen_triples(A)
    if A.format == 'dok':
        return _dok_gen_triples(A)
    return _coo_gen_triples(A.tocoo())