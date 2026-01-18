from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify
from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.iterables import (has_dups, sift, iterable,
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name
import mpmath
from mpmath.libmp.libmpf import prec_to_dps
import inspect
from collections import Counter
from .symbol import Dummy, Symbol
class WildFunction(Function, AtomicExpr):
    """
    A WildFunction function matches any function (with its arguments).

    Examples
    ========

    >>> from sympy import WildFunction, Function, cos
    >>> from sympy.abc import x, y
    >>> F = WildFunction('F')
    >>> f = Function('f')
    >>> F.nargs
    Naturals0
    >>> x.match(F)
    >>> F.match(F)
    {F_: F_}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> cos(x).match(F)
    {F_: cos(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a given number of arguments, set ``nargs`` to the
    desired value at instantiation:

    >>> F = WildFunction('F', nargs=2)
    >>> F.nargs
    {2}
    >>> f(x).match(F)
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a range of arguments, set ``nargs`` to a tuple
    containing the desired number of arguments, e.g. if ``nargs = (1, 2)``
    then functions with 1 or 2 arguments will be matched.

    >>> F = WildFunction('F', nargs=(1, 2))
    >>> F.nargs
    {1, 2}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}
    >>> f(x, y, 1).match(F)

    """
    include: set[Any] = set()

    def __init__(cls, name, **assumptions):
        from sympy.sets.sets import Set, FiniteSet
        cls.name = name
        nargs = assumptions.pop('nargs', S.Naturals0)
        if not isinstance(nargs, Set):
            if is_sequence(nargs):
                nargs = tuple(ordered(set(nargs)))
            elif nargs is not None:
                nargs = (as_int(nargs),)
            nargs = FiniteSet(*nargs)
        cls.nargs = nargs

    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, (AppliedUndef, Function)):
            return None
        if len(expr.args) not in self.nargs:
            return None
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()
        repl_dict[self] = expr
        return repl_dict