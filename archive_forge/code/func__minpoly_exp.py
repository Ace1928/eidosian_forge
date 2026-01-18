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
def _minpoly_exp(ex, x):
    """
    Returns the minimal polynomial of ``exp(ex)``
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a == I * pi:
        if c.is_rational:
            q = sympify(c.q)
            if c.p == 1 or c.p == -1:
                if q == 3:
                    return x ** 2 - x + 1
                if q == 4:
                    return x ** 4 + 1
                if q == 6:
                    return x ** 4 - x ** 2 + 1
                if q == 8:
                    return x ** 8 + 1
                if q == 9:
                    return x ** 6 - x ** 3 + 1
                if q == 10:
                    return x ** 8 - x ** 6 + x ** 4 - x ** 2 + 1
                if q.is_prime:
                    s = 0
                    for i in range(q):
                        s += (-x) ** i
                    return s
            factors = [cyclotomic_poly(i, x) for i in divisors(2 * q)]
            mp = _choose_factor(factors, x, ex)
            return mp
        else:
            raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)
    raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)