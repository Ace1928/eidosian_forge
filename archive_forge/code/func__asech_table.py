from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_or, fuzzy_and, FuzzyBool
from sympy.core.numbers import I, pi, Rational
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import (binomial, factorial,
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
from sympy.polys.specialpolys import symmetric_poly
@cacheit
def _asech_table():
    return {I: -(pi * I / 2) + log(1 + sqrt(2)), -I: pi * I / 2 + log(1 + sqrt(2)), sqrt(6) - sqrt(2): pi / 12, sqrt(2) - sqrt(6): 11 * pi / 12, sqrt(2 - 2 / sqrt(5)): pi / 10, -sqrt(2 - 2 / sqrt(5)): 9 * pi / 10, 2 / sqrt(2 + sqrt(2)): pi / 8, -2 / sqrt(2 + sqrt(2)): 7 * pi / 8, 2 / sqrt(3): pi / 6, -2 / sqrt(3): 5 * pi / 6, sqrt(5) - 1: pi / 5, 1 - sqrt(5): 4 * pi / 5, sqrt(2): pi / 4, -sqrt(2): 3 * pi / 4, sqrt(2 + 2 / sqrt(5)): 3 * pi / 10, -sqrt(2 + 2 / sqrt(5)): 7 * pi / 10, S(2): pi / 3, -S(2): 2 * pi / 3, sqrt(2 * (2 + sqrt(2))): 3 * pi / 8, -sqrt(2 * (2 + sqrt(2))): 5 * pi / 8, 1 + sqrt(5): 2 * pi / 5, -1 - sqrt(5): 3 * pi / 5, sqrt(6) + sqrt(2): 5 * pi / 12, -sqrt(6) - sqrt(2): 7 * pi / 12, I * S.Infinity: -pi * I / 2, I * S.NegativeInfinity: pi * I / 2}