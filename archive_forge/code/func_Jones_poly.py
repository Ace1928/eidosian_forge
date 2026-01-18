from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
import sage.graphs.graph as graph
from sage.rings.rational_field import QQ
def Jones_poly(K, variable=None, new_convention=False):
    """
    The old convention should really have powers of q^(1/2) for links
    with an odd number of components, but it just multiplies the
    answer by q^(1/2) to get rid of them.  Moroever, the choice of
    value for the unlink is a little screwy, essentially::

      (-q^(1/2) - q^(-1/2))^(n - 1).

    In the new convention, powers of q^(1/2) never appear, i.e. the
    new q is the old q^(1/2) and moreover the value for an n-component
    unlink is (q + 1/q)^(n - 1).  This should match Bar-Natan's paper
    on Khovanov homology.
    """
    if not variable:
        L = LaurentPolynomialRing(QQ, 'q')
        variable = L.gen()
    answer = 0
    L_A = LaurentPolynomialRing(QQ, 'A')
    A = L_A.gen()
    G = K.white_graph()
    for i, labels in enumerate(G.edge_labels()):
        labels['edge_index'] = i
    writhe = K.writhe()
    for T in spanning_trees(G):
        answer = answer + _Jones_contrib(K, G, T, A)
    answer = answer * (-A) ** (3 * writhe)
    ans = 0
    for i in range(len(answer.coefficients())):
        coeff = answer.coefficients()[i]
        exp = answer.exponents()[i]
        if new_convention:
            assert exp % 2 == 0
            ans = ans + coeff * (-variable) ** (exp // 2)
        else:
            ans = ans + coeff * variable ** (exp // 4)
    return ans