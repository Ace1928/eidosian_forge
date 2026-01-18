from sympy.polys.monomials import monomial_mul, monomial_div
def _incr_k(m, k):
    return tuple(list(m[:k]) + [m[k] + 1] + list(m[k + 1:]))