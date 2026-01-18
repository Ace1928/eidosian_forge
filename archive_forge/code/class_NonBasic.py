from sympy.assumptions.refine import refine
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import (ExprBuilder, unchanged, Expr,
from sympy.core.function import (Function, expand, WildFunction,
from sympy.core.mul import Mul
from sympy.core.numbers import (NumberSymbol, E, zoo, oo, Float, I,
from sympy.core.power import Pow
from sympy.core.relational import Ge, Lt, Gt, Le
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols, Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import tan, sin, cos
from sympy.functions.special.delta_functions import (Heaviside,
from sympy.functions.special.error_functions import Si
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate, Integral
from sympy.physics.secondquant import FockState
from sympy.polys.partfrac import apart
from sympy.polys.polytools import factor, cancel, Poly
from sympy.polys.rationaltools import together
from sympy.series.order import O
from sympy.sets.sets import FiniteSet
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import collect, radsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify, nsimplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import Indexed
from sympy.physics.units import meter
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import a, b, c, n, t, u, x, y, z
class NonBasic:
    """This class represents an object that knows how to implement binary
    operations like +, -, etc with Expr but is not a subclass of Basic itself.
    The NonExpr subclass below does subclass Basic but not Expr.

    For both NonBasic and NonExpr it should be possible for them to override
    Expr.__add__ etc because Expr.__add__ should be returning NotImplemented
    for non Expr classes. Otherwise Expr.__add__ would create meaningless
    objects like Add(Integer(1), FiniteSet(2)) and it wouldn't be possible for
    other classes to override these operations when interacting with Expr.
    """

    def __add__(self, other):
        return SpecialOp('+', self, other)

    def __radd__(self, other):
        return SpecialOp('+', other, self)

    def __sub__(self, other):
        return SpecialOp('-', self, other)

    def __rsub__(self, other):
        return SpecialOp('-', other, self)

    def __mul__(self, other):
        return SpecialOp('*', self, other)

    def __rmul__(self, other):
        return SpecialOp('*', other, self)

    def __truediv__(self, other):
        return SpecialOp('/', self, other)

    def __rtruediv__(self, other):
        return SpecialOp('/', other, self)

    def __floordiv__(self, other):
        return SpecialOp('//', self, other)

    def __rfloordiv__(self, other):
        return SpecialOp('//', other, self)

    def __mod__(self, other):
        return SpecialOp('%', self, other)

    def __rmod__(self, other):
        return SpecialOp('%', other, self)

    def __divmod__(self, other):
        return SpecialOp('divmod', self, other)

    def __rdivmod__(self, other):
        return SpecialOp('divmod', other, self)

    def __pow__(self, other):
        return SpecialOp('**', self, other)

    def __rpow__(self, other):
        return SpecialOp('**', other, self)

    def __lt__(self, other):
        return SpecialOp('<', self, other)

    def __gt__(self, other):
        return SpecialOp('>', self, other)

    def __le__(self, other):
        return SpecialOp('<=', self, other)

    def __ge__(self, other):
        return SpecialOp('>=', self, other)