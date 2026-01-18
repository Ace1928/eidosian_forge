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
def _get_examples_ode_sol_1st_homogeneous_coeff_best():
    return {'hint': '1st_homogeneous_coeff_best', 'func': f(x), 'examples': {'1st_homogeneous_coeff_best_01': {'eq': f(x) + (x * log(f(x) / x) - 2 * x) * diff(f(x), x), 'sol': [Eq(f(x), -exp(C1) * LambertW(-x * exp(-C1 + 1)))], 'checkodesol_XFAIL': True}, '1st_homogeneous_coeff_best_02': {'eq': 2 * f(x) * exp(x / f(x)) + f(x) * f(x).diff(x) - 2 * x * exp(x / f(x)) * f(x).diff(x), 'sol': [Eq(log(f(x)), C1 - 2 * exp(x / f(x)))]}, '1st_homogeneous_coeff_best_03': {'eq': 2 * x ** 2 * f(x) + f(x) ** 3 + (x * f(x) ** 2 - 2 * x ** 3) * f(x).diff(x), 'sol': [Eq(f(x), exp(2 * C1 + LambertW(-2 * x ** 4 * exp(-4 * C1)) / 2) / x)], 'checkodesol_XFAIL': True}, '1st_homogeneous_coeff_best_04': {'eq': (x + sqrt(f(x) ** 2 - x * f(x))) * f(x).diff(x) - f(x), 'sol': [Eq(log(f(x)), C1 - 2 * sqrt(-x / f(x) + 1))], 'slow': True}, '1st_homogeneous_coeff_best_05': {'eq': x + f(x) - (x - f(x)) * f(x).diff(x), 'sol': [Eq(log(x), C1 - log(sqrt(1 + f(x) ** 2 / x ** 2)) + atan(f(x) / x))]}, '1st_homogeneous_coeff_best_06': {'eq': x * f(x).diff(x) - f(x) - x * sin(f(x) / x), 'sol': [Eq(f(x), 2 * x * atan(C1 * x))]}, '1st_homogeneous_coeff_best_07': {'eq': x ** 2 + f(x) ** 2 - 2 * x * f(x) * f(x).diff(x), 'sol': [Eq(f(x), -sqrt(x * (C1 + x))), Eq(f(x), sqrt(x * (C1 + x)))]}, '1st_homogeneous_coeff_best_08': {'eq': f(x) ** 2 + (x * sqrt(f(x) ** 2 - x ** 2) - x * f(x)) * f(x).diff(x), 'sol': [Eq(f(x), -sqrt(-x * exp(2 * C1) / (x - 2 * exp(C1)))), Eq(f(x), sqrt(-x * exp(2 * C1) / (x - 2 * exp(C1))))], 'checkodesol_XFAIL': True}}}