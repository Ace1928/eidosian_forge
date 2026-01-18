from sympy.core.function import (Derivative, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (Ei, erfi)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.rootoftools import rootof
from sympy.core import Function, Symbol
from sympy.functions import airyai, airybi, besselj, bessely, lowergamma
from sympy.integrals.risch import NonElementaryIntegral
from sympy.solvers.ode import classify_ode, dsolve
from sympy.solvers.ode.ode import allhints, _remove_redundant_solutions
from sympy.solvers.ode.single import (FirstLinear, ODEMatchError,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import raises, slow, ON_CI
import traceback
from sympy.solvers.ode.tests.test_single import _test_an_example
@_add_example_keys
def _get_examples_ode_sol_1st_rational_riccati():
    return {'hint': '1st_rational_riccati', 'func': f(x), 'examples': {'rational_riccati_01': {'eq': Eq(f(x).diff(x) + f(x) ** 2 - 2, 0), 'sol': [Eq(f(x), sqrt(2) * (-C1 - exp(2 * sqrt(2) * x)) / (C1 - exp(2 * sqrt(2) * x)))]}, 'rational_riccati_02': {'eq': f(x) ** 2 + Derivative(f(x), x) + 4 * f(x) / x + 2 / x ** 2, 'sol': [Eq(f(x), (-2 * C1 - x) / (x * (C1 + x)))]}, 'rational_riccati_03': {'eq': 2 * x ** 2 * Derivative(f(x), x) - x * (4 * f(x) + Derivative(f(x), x) - 4) + (f(x) - 1) * f(x), 'sol': [Eq(f(x), (C1 + 2 * x ** 2) / (C1 + x))]}, 'rational_riccati_04': {'eq': f(x).diff(x) - 6 - 5 * f(x) - f(x) ** 2, 'sol': [Eq(f(x), (-2 * C1 + 3 * exp(x)) / (C1 - exp(x)))]}, 'rational_riccati_05': {'eq': x ** 2 - (2 * x + 1 / x) * f(x) + f(x) ** 2 + Derivative(f(x), x), 'sol': [Eq(f(x), x * (C1 + x ** 2 + 1) / (C1 + x ** 2 - 1))]}, 'rational_riccati_06': {'eq': x ** 4 * Derivative(f(x), x) + x ** 2 - x * (2 * f(x) ** 2 + Derivative(f(x), x)) + f(x), 'sol': [Eq(f(x), x * (C1 * x - x + 1) / (C1 + x ** 2 - 1))]}, 'rational_riccati_07': {'eq': -f(x) ** 2 + Derivative(f(x), x) + (15 * x ** 2 - 20 * x + 7) / ((x - 1) ** 2 * (2 * x - 1) ** 2), 'sol': [Eq(f(x), (9 * C1 * x - 6 * C1 - 15 * x ** 5 + 60 * x ** 4 - 94 * x ** 3 + 72 * x ** 2 - 33 * x + 8) / (6 * C1 * x ** 2 - 9 * C1 * x + 3 * C1 + 6 * x ** 6 - 29 * x ** 5 + 57 * x ** 4 - 58 * x ** 3 + 28 * x ** 2 - 3 * x - 1))]}, 'rational_riccati_08': {'eq': Derivative(f(x), x) + (3 * x ** 2 + 1) * f(x) ** 2 / x + (6 * x ** 2 - x + 3) * f(x) / (x * (x - 1)) + (3 * x ** 2 - 2 * x + 2) / (x * (x - 1) ** 2), 'sol': [Eq(f(x), (-C1 - x ** 3 + x ** 2 - 2 * x + 1) / (C1 * x - C1 + x ** 4 - x ** 3 + x ** 2 - 2 * x + 1))]}, 'rational_riccati_09': {'eq': Derivative(f(x), x) - 2 * I * (f(x) ** 2 + 1) / x, 'sol': [Eq(f(x), (-I * C1 + I * x ** 4 + I) / (C1 + x ** 4 - 1))]}, 'rational_riccati_10': {'eq': Eq(Derivative(f(x), x), x * f(x) / (S(3) / 2 - 2 * x) + (x / 2 - S(1) / 3) * f(x) ** 2 / (2 * x / 3 - S(1) / 2) - S(5) / 4 + (281 * x ** 2 - 1260 * x + 756) / (16 * x ** 3 - 12 * x ** 2)), 'sol': [Eq(f(x), (40 * C1 * x ** 14 + 28 * C1 * x ** 13 + 420 * C1 * x ** 12 + 2940 * C1 * x ** 11 + 18480 * C1 * x ** 10 + 103950 * C1 * x ** 9 + 519750 * C1 * x ** 8 + 2286900 * C1 * x ** 7 + 8731800 * C1 * x ** 6 + 28378350 * C1 * x ** 5 + 76403250 * C1 * x ** 4 + 163721250 * C1 * x ** 3 + 261954000 * C1 * x ** 2 + 278326125 * C1 * x + 147349125 * C1 + x * exp(2 * x) - 9 * exp(2 * x)) / (x * (24 * C1 * x ** 13 + 140 * C1 * x ** 12 + 840 * C1 * x ** 11 + 4620 * C1 * x ** 10 + 23100 * C1 * x ** 9 + 103950 * C1 * x ** 8 + 415800 * C1 * x ** 7 + 1455300 * C1 * x ** 6 + 4365900 * C1 * x ** 5 + 10914750 * C1 * x ** 4 + 21829500 * C1 * x ** 3 + 32744250 * C1 * x ** 2 + 32744250 * C1 * x + 16372125 * C1 - exp(2 * x))))]}}}