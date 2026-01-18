from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, Function
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly
@cacheit
def _dilogtable():
    return {S.Half: pi ** 2 / 12 - log(2) ** 2 / 2, Integer(2): pi ** 2 / 4 - I * pi * log(2), -(sqrt(5) - 1) / 2: -pi ** 2 / 15 + log((sqrt(5) - 1) / 2) ** 2 / 2, -(sqrt(5) + 1) / 2: -pi ** 2 / 10 - log((sqrt(5) + 1) / 2) ** 2, (3 - sqrt(5)) / 2: pi ** 2 / 15 - log((sqrt(5) - 1) / 2) ** 2, (sqrt(5) - 1) / 2: pi ** 2 / 10 - log((sqrt(5) - 1) / 2) ** 2, I: I * S.Catalan - pi ** 2 / 48, -I: -I * S.Catalan - pi ** 2 / 48, 1 - I: pi ** 2 / 16 - I * S.Catalan - pi * I / 4 * log(2), 1 + I: pi ** 2 / 16 + I * S.Catalan + pi * I / 4 * log(2), (1 - I) / 2: -log(2) ** 2 / 8 + pi * I * log(2) / 8 + 5 * pi ** 2 / 96 - I * S.Catalan}