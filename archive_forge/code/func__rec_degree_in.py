from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def _rec_degree_in(g, v, i, j):
    """Recursive helper function for :func:`dmp_degree_in`."""
    if i == j:
        return dmp_degree(g, v)
    v, i = (v - 1, i + 1)
    return max([_rec_degree_in(c, v, i, j) for c in g])