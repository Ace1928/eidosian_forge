from functools import reduce
from itertools import product
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import oo, igcd, Rational
from sympy.core.relational import Eq, is_eq
from sympy.core.kind import NumberKind
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.logic.boolalg import And, Or
from .sets import Set, Interval, Union, FiniteSet, ProductSet, SetKind
from sympy.utilities.misc import filldedent
class Rationals(Set, metaclass=Singleton):
    """
    Represents the rational numbers. This set is also available as
    the singleton ``S.Rationals``.

    Examples
    ========

    >>> from sympy import S
    >>> S.Half in S.Rationals
    True
    >>> iterable = iter(S.Rationals)
    >>> [next(iterable) for i in range(12)]
    [0, 1, -1, 1/2, 2, -1/2, -2, 1/3, 3, -1/3, -3, 2/3]
    """
    is_iterable = True
    _inf = S.NegativeInfinity
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return False
        return other.is_rational

    def __iter__(self):
        yield S.Zero
        yield S.One
        yield S.NegativeOne
        d = 2
        while True:
            for n in range(d):
                if igcd(n, d) == 1:
                    yield Rational(n, d)
                    yield Rational(d, n)
                    yield Rational(-n, d)
                    yield Rational(-d, n)
            d += 1

    @property
    def _boundary(self):
        return S.Reals

    def _kind(self):
        return SetKind(NumberKind)