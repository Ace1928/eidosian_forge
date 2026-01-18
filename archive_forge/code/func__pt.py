import itertools
from sympy.calculus.util import (continuous_domain, periodicity,
from sympy.core import Symbol, Dummy, sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Relational, Eq, Ge, Lt
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection
from sympy.core.singleton import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.complexes import im, Abs
from sympy.logic import And
from sympy.polys import Poly, PolynomialError, parallel_poly_from_expr
from sympy.polys.polyutils import _nsort
from sympy.solvers.solveset import solvify, solveset
from sympy.utilities.iterables import sift, iterable
from sympy.utilities.misc import filldedent
def _pt(start, end):
    """Return a point between start and end"""
    if not start.is_infinite and (not end.is_infinite):
        pt = (start + end) / 2
    elif start.is_infinite and end.is_infinite:
        pt = S.Zero
    else:
        if start.is_infinite and start.is_extended_positive is None or (end.is_infinite and end.is_extended_positive is None):
            raise ValueError('cannot proceed with unsigned infinite values')
        if end.is_infinite and end.is_extended_negative or (start.is_infinite and start.is_extended_positive):
            start, end = (end, start)
        if end.is_infinite:
            if start.is_extended_positive:
                pt = start * 2
            elif start.is_extended_negative:
                pt = start * S.Half
            else:
                pt = start + 1
        elif start.is_infinite:
            if end.is_extended_positive:
                pt = end * S.Half
            elif end.is_extended_negative:
                pt = end * 2
            else:
                pt = end - 1
    return pt