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
class UnShiftA(Operator):
    """ Decrement an upper index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        ap, bq, i = list(map(sympify, [ap, bq, i]))
        self._ap = ap
        self._bq = bq
        self._i = i
        ap = list(ap)
        bq = list(bq)
        ai = ap.pop(i) - 1
        if ai == 0:
            raise ValueError('Cannot decrement unit upper index.')
        m = Poly(z * ai, _x)
        for a in ap:
            m *= Poly(_x + a, _x)
        A = Dummy('A')
        n = D = Poly(ai * A - ai, A)
        for b in bq:
            n *= D + (b - 1).as_poly(A)
        b0 = -n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper index: cancels with lower')
        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, _x / ai + 1), _x)
        self._poly = Poly((n - m) / b0, _x)

    def __str__(self):
        return '<Decrement upper index #%s of %s, %s.>' % (self._i, self._ap, self._bq)