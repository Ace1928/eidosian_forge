from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def _rec_degree_list(g, v, i, degs):
    """Recursive helper for :func:`dmp_degree_list`."""
    degs[i] = max(degs[i], dmp_degree(g, v))
    if v > 0:
        v, i = (v - 1, i + 1)
        for c in g:
            _rec_degree_list(c, v, i, degs)