from typing import Tuple as tTuple, Union as tUnion
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError, expand_mul
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.relational import Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.combinatorial.numbers import bernoulli, euler
from sympy.functions.elementary.complexes import arg as arg_f, im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary._trigonometric_special import (
from sympy.logic.boolalg import And
from sympy.ntheory import factorint
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.iterables import numbered_symbols
@staticmethod
@cacheit
def _acsc_table():
    return {2 * sqrt(3) / 3: pi / 3, sqrt(2): pi / 4, sqrt(2 + 2 * sqrt(5) / 5): pi / 5, 1 / sqrt(Rational(5, 8) - sqrt(5) / 8): pi / 5, sqrt(2 - 2 * sqrt(5) / 5): pi * Rational(2, 5), 1 / sqrt(Rational(5, 8) + sqrt(5) / 8): pi * Rational(2, 5), 2: pi / 6, sqrt(4 + 2 * sqrt(2)): pi / 8, 2 / sqrt(2 - sqrt(2)): pi / 8, sqrt(4 - 2 * sqrt(2)): pi * Rational(3, 8), 2 / sqrt(2 + sqrt(2)): pi * Rational(3, 8), 1 + sqrt(5): pi / 10, sqrt(5) - 1: pi * Rational(3, 10), -(sqrt(5) - 1): pi * Rational(-3, 10), sqrt(6) + sqrt(2): pi / 12, sqrt(6) - sqrt(2): pi * Rational(5, 12), -(sqrt(6) - sqrt(2)): pi * Rational(-5, 12)}