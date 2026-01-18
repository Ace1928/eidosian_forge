from sympy.polys.partfrac import (
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x, y, a, b, c
def _make_extension_example():
    from sympy.core import Mul

    def mul2(expr):
        return Mul(2, expr, evaluate=False)
    f = (x ** 2 + 1) ** 3 / ((x - 1) ** 2 * (x + 1) ** 2 * (-x ** 2 + 2 * x + 1) * (x ** 2 + 2 * x - 1))
    g = 1 / mul2(x - sqrt(2) + 1) - 1 / mul2(x - sqrt(2) - 1) + 1 / mul2(x + 1 + sqrt(2)) - 1 / mul2(x - 1 + sqrt(2)) + 1 / mul2((x + 1) ** 2) + 1 / mul2((x - 1) ** 2)
    return (f, g)