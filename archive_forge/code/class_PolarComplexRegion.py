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
class PolarComplexRegion(ComplexRegion):
    """
    Set representing a polar region of the complex plane.

    .. math:: Z = \\{z \\in \\mathbb{C} \\mid z = r\\times (\\cos(\\theta) + I\\sin(\\theta)), r \\in [\\texttt{r}], \\theta \\in [\\texttt{theta}]\\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, oo, pi, I
    >>> rset = Interval(0, oo)
    >>> thetaset = Interval(0, pi)
    >>> upper_half_plane = ComplexRegion(rset * thetaset, polar=True)
    >>> 1 + I in upper_half_plane
    True
    >>> 1 - I in upper_half_plane
    False

    See also
    ========

    ComplexRegion
    CartesianComplexRegion
    Complexes

    """
    polar = True
    variables = symbols('r, theta', cls=Dummy)

    def __new__(cls, sets):
        new_sets = []
        if not sets.is_ProductSet:
            for k in sets.args:
                new_sets.append(k)
        else:
            new_sets.append(sets)
        for k, v in enumerate(new_sets):
            new_sets[k] = ProductSet(v.args[0], normalize_theta_set(v.args[1]))
        sets = Union(*new_sets)
        return Set.__new__(cls, sets)

    @property
    def expr(self):
        r, theta = self.variables
        return r * (cos(theta) + S.ImaginaryUnit * sin(theta))