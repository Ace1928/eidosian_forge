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
Find the equations relating symbols in expr and candidate.