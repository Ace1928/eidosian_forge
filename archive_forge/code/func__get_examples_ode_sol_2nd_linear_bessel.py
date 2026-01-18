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
def _get_examples_ode_sol_2nd_linear_bessel():
    return {'hint': '2nd_linear_bessel', 'func': f(x), 'examples': {'2nd_lin_bessel_01': {'eq': x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + (x ** 2 - 4) * f(x), 'sol': [Eq(f(x), C1 * besselj(2, x) + C2 * bessely(2, x))]}, '2nd_lin_bessel_02': {'eq': x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + (x ** 2 + 25) * f(x), 'sol': [Eq(f(x), C1 * besselj(5 * I, x) + C2 * bessely(5 * I, x))]}, '2nd_lin_bessel_03': {'eq': x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + x ** 2 * f(x), 'sol': [Eq(f(x), C1 * besselj(0, x) + C2 * bessely(0, x))]}, '2nd_lin_bessel_04': {'eq': x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + (81 * x ** 2 - S(1) / 9) * f(x), 'sol': [Eq(f(x), C1 * besselj(S(1) / 3, 9 * x) + C2 * bessely(S(1) / 3, 9 * x))]}, '2nd_lin_bessel_05': {'eq': x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + (x ** 4 - 4) * f(x), 'sol': [Eq(f(x), C1 * besselj(1, x ** 2 / 2) + C2 * bessely(1, x ** 2 / 2))]}, '2nd_lin_bessel_06': {'eq': x ** 2 * f(x).diff(x, 2) + 2 * x * f(x).diff(x) + (x ** 4 - 4) * f(x), 'sol': [Eq(f(x), (C1 * besselj(sqrt(17) / 4, x ** 2 / 2) + C2 * bessely(sqrt(17) / 4, x ** 2 / 2)) / sqrt(x))]}, '2nd_lin_bessel_07': {'eq': x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + (x ** 2 - S(1) / 4) * f(x), 'sol': [Eq(f(x), C1 * besselj(S(1) / 2, x) + C2 * bessely(S(1) / 2, x))]}, '2nd_lin_bessel_08': {'eq': x ** 2 * f(x).diff(x, 2) - 3 * x * f(x).diff(x) + (4 * x + 4) * f(x), 'sol': [Eq(f(x), x ** 2 * (C1 * besselj(0, 4 * sqrt(x)) + C2 * bessely(0, 4 * sqrt(x))))]}, '2nd_lin_bessel_09': {'eq': x * f(x).diff(x, 2) - f(x).diff(x) + 4 * x ** 3 * f(x), 'sol': [Eq(f(x), x * (C1 * besselj(S(1) / 2, x ** 2) + C2 * bessely(S(1) / 2, x ** 2)))]}, '2nd_lin_bessel_10': {'eq': (x - 2) ** 2 * f(x).diff(x, 2) - (x - 2) * f(x).diff(x) + 4 * (x - 2) ** 2 * f(x), 'sol': [Eq(f(x), (x - 2) * (C1 * besselj(1, 2 * x - 4) + C2 * bessely(1, 2 * x - 4)))]}, '2nd_lin_bessel_11': {'eq': f(x).diff(x, x) + 2 / x * f(x).diff(x) + f(x), 'sol': [Eq(f(x), (C1 * besselj(S(1) / 2, x) + C2 * bessely(S(1) / 2, x)) / sqrt(x))]}}}