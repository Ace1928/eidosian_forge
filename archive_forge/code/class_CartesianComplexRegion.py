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
class CartesianComplexRegion(ComplexRegion):
    """
    Set representing a square region of the complex plane.

    .. math:: Z = \\{z \\in \\mathbb{C} \\mid z = x + Iy, x \\in [\\operatorname{re}(z)], y \\in [\\operatorname{im}(z)]\\}

    Examples
    ========

    >>> from sympy import ComplexRegion, I, Interval
    >>> region = ComplexRegion(Interval(1, 3) * Interval(4, 6))
    >>> 2 + 5*I in region
    True
    >>> 5*I in region
    False

    See also
    ========

    ComplexRegion
    PolarComplexRegion
    Complexes
    """
    polar = False
    variables = symbols('x, y', cls=Dummy)

    def __new__(cls, sets):
        if sets == S.Reals * S.Reals:
            return S.Complexes
        if all((_a.is_FiniteSet for _a in sets.args)) and len(sets.args) == 2:
            complex_num = []
            for x in sets.args[0]:
                for y in sets.args[1]:
                    complex_num.append(x + S.ImaginaryUnit * y)
            return FiniteSet(*complex_num)
        else:
            return Set.__new__(cls, sets)

    @property
    def expr(self):
        x, y = self.variables
        return x + S.ImaginaryUnit * y