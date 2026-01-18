from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def _rec_inflate(g, M, v, i, K):
    """Recursive helper for :func:`dmp_inflate`."""
    if not v:
        return dup_inflate(g, M[i], K)
    if M[i] <= 0:
        raise IndexError('all M[i] must be positive, got %s' % M[i])
    w, j = (v - 1, i + 1)
    g = [_rec_inflate(c, M, w, j, K) for c in g]
    result = [g[0]]
    for coeff in g[1:]:
        for _ in range(1, M[i]):
            result.append(dmp_zero(w))
        result.append(coeff)
    return result