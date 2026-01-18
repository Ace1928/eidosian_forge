from functools import reduce
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.core.function import expand_mul, expand_multinomial, _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi, _illegal)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.ntheory.factor_ import divisors
from sympy.utilities.iterables import subsets
from sympy.polys.domains import ZZ, QQ, FractionField
from sympy.polys.orthopolys import dup_chebyshevt
from sympy.polys.polyerrors import (
from sympy.polys.polytools import (
from sympy.polys.polyutils import dict_from_expr, expr_from_dict
from sympy.polys.ring_series import rs_compose_add
from sympy.polys.rings import ring
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import (
def _choose_factor(factors, x, v, dom=QQ, prec=200, bound=5):
    """
    Return a factor having root ``v``
    It is assumed that one of the factors has root ``v``.
    """
    if isinstance(factors[0], tuple):
        factors = [f[0] for f in factors]
    if len(factors) == 1:
        return factors[0]
    prec1 = 10
    points = {}
    symbols = dom.symbols if hasattr(dom, 'symbols') else []
    while prec1 <= prec:
        fe = [f.as_expr().xreplace({x: v}) for f in factors]
        if v.is_number:
            fe = [f.n(prec) for f in fe]
        for n in subsets(range(bound), k=len(symbols), repetition=True):
            for s, i in zip(symbols, n):
                points[s] = i
            candidates = [(abs(f.subs(points).n(prec1)), i) for i, f in enumerate(fe)]
            if any((i in _illegal for i, _ in candidates)):
                continue
            can = sorted(candidates)
            (a, ix), (b, _) = can[:2]
            if b > a * 10 ** 6:
                return factors[ix]
        prec1 *= 2
    raise NotImplementedError('multiple candidates for the minimal polynomial of %s' % v)