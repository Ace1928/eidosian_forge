import itertools
from sympy.core import S
from sympy.core.sorting import default_sort_key
from sympy.polys import Poly, groebner, roots
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.polys.polyerrors import (ComputationFailed,
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.misc import filldedent
def _subs_root(f, gen, zero):
    """Replace generator with a root so that the result is nice. """
    p = f.as_expr({gen: zero})
    if f.degree(gen) >= 2:
        p = p.expand(deep=False)
    return p