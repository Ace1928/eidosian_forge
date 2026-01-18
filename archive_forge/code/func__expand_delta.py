from .products import product
from .summations import Sum, summation
from sympy.core import Add, Mul, S, Dummy
from sympy.core.cache import cacheit
from sympy.core.sorting import default_sort_key
from sympy.functions import KroneckerDelta, Piecewise, piecewise_fold
from sympy.polys.polytools import factor
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve
@cacheit
def _expand_delta(expr, index):
    """
    Expand the first Add containing a simple KroneckerDelta.
    """
    if not expr.is_Mul:
        return expr
    delta = None
    func = Add
    terms = [S.One]
    for h in expr.args:
        if delta is None and h.is_Add and _has_simple_delta(h, index):
            delta = True
            func = h.func
            terms = [terms[0] * t for t in h.args]
        else:
            terms = [t * h for t in terms]
    return func(*terms)