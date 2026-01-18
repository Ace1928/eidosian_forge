from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
@classmethod
def _meijer(cls, b, a, sign):
    """ Cancel b + sign*s and a + sign*s
            This is for meijer G functions. """
    b = sympify(b)
    a = sympify(a)
    n = b - a
    if n.is_negative or not n.is_Integer:
        return None
    expr = Operator.__new__(cls)
    p = S.One
    for k in range(n):
        p *= sign * _x + a + k
    expr._poly = Poly(p, _x)
    if sign == -1:
        expr._a = b
        expr._b = a
    else:
        expr._b = Add(1, a - 1, evaluate=False)
        expr._a = Add(1, b - 1, evaluate=False)
    return expr