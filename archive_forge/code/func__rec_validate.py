from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def _rec_validate(f, g, i, K):
    """Recursive helper for :func:`dmp_validate`."""
    if not isinstance(g, list):
        if K is not None and (not K.of_type(g)):
            raise TypeError('%s in %s in not of type %s' % (g, f, K.dtype))
        return {i - 1}
    elif not g:
        return {i}
    else:
        levels = set()
        for c in g:
            levels |= _rec_validate(f, c, i + 1, K)
        return levels