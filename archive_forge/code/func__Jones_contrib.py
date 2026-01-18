from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
import sage.graphs.graph as graph
from sage.rings.rational_field import QQ
def _Jones_contrib(K, G, T, A):
    """Returns the contribution to the Jones polynomial of the tree T. G should be self.black_graph()."""
    answer = 1
    for e in G.edges(sort=True, key=edge_index):
        answer = answer * _Jones_contrib_edge(K, G, T, e, A)
    return answer