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
class ReduceOrder(Operator):
    """ Reduce Order by cancelling an upper and a lower index. """

    def __new__(cls, ai, bj):
        """ For convenience if reduction is not possible, return None. """
        ai = sympify(ai)
        bj = sympify(bj)
        n = ai - bj
        if not n.is_Integer or n < 0:
            return None
        if bj.is_integer and bj.is_nonpositive:
            return None
        expr = Operator.__new__(cls)
        p = S.One
        for k in range(n):
            p *= (_x + bj + k) / (bj + k)
        expr._poly = Poly(p, _x)
        expr._a = ai
        expr._b = bj
        return expr

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

    @classmethod
    def meijer_minus(cls, b, a):
        return cls._meijer(b, a, -1)

    @classmethod
    def meijer_plus(cls, a, b):
        return cls._meijer(1 - a, 1 - b, 1)

    def __str__(self):
        return '<Reduce order by cancelling upper %s with lower %s.>' % (self._a, self._b)