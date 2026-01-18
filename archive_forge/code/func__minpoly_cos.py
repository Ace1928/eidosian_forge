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
def _minpoly_cos(ex, x):
    """
    Returns the minimal polynomial of ``cos(ex)``
    see https://mathworld.wolfram.com/TrigonometryAngles.html
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            if c.p == 1:
                if c.q == 7:
                    return 8 * x ** 3 - 4 * x ** 2 - 4 * x + 1
                if c.q == 9:
                    return 8 * x ** 3 - 6 * x - 1
            elif c.p == 2:
                q = sympify(c.q)
                if q.is_prime:
                    s = _minpoly_sin(ex, x)
                    return _mexpand(s.subs({x: sqrt((1 - x) / 2)}))
            n = int(c.q)
            a = dup_chebyshevt(n, ZZ)
            a = [x ** (n - i) * a[i] for i in range(n + 1)]
            r = Add(*a) - (-1) ** c.p
            _, factors = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res
    raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)