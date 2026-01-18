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
def _get_examples_ode_sol_euler_undetermined_coeff():
    return {'hint': 'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients', 'func': f(x), 'examples': {'euler_undet_01': {'eq': Eq(x ** 2 * diff(f(x), x, x) + x * diff(f(x), x), 1), 'sol': [Eq(f(x), C1 + C2 * log(x) + log(x) ** 2 / 2)]}, 'euler_undet_02': {'eq': Eq(x ** 2 * diff(f(x), x, x) - 2 * x * diff(f(x), x) + 2 * f(x), x ** 3), 'sol': [Eq(f(x), x * (C1 + C2 * x + Rational(1, 2) * x ** 2))]}, 'euler_undet_03': {'eq': Eq(x ** 2 * diff(f(x), x, x) - x * diff(f(x), x) - 3 * f(x), log(x) / x), 'sol': [Eq(f(x), (C1 + C2 * x ** 4 - log(x) ** 2 / 8 - log(x) / 16) / x)]}, 'euler_undet_04': {'eq': Eq(x ** 2 * diff(f(x), x, x) + 3 * x * diff(f(x), x) - 8 * f(x), log(x) ** 3 - log(x)), 'sol': [Eq(f(x), C1 / x ** 4 + C2 * x ** 2 - Rational(1, 8) * log(x) ** 3 - Rational(3, 32) * log(x) ** 2 - Rational(1, 64) * log(x) - Rational(7, 256))]}, 'euler_undet_05': {'eq': Eq(x ** 3 * diff(f(x), x, x, x) - 3 * x ** 2 * diff(f(x), x, x) + 6 * x * diff(f(x), x) - 6 * f(x), log(x)), 'sol': [Eq(f(x), C1 * x + C2 * x ** 2 + C3 * x ** 3 - Rational(1, 6) * log(x) - Rational(11, 36))]}, 'euler_undet_06': {'eq': 2 * x ** 2 * f(x).diff(x, 2) + f(x) + sqrt(2 * x) * sin(log(2 * x) / 2), 'sol': [Eq(f(x), sqrt(x) * (C1 * sin(log(x) / 2) + C2 * cos(log(x) / 2) + sqrt(2) * log(x) * cos(log(2 * x) / 2) / 2))]}, 'euler_undet_07': {'eq': 2 * x ** 2 * f(x).diff(x, 2) + f(x) + sin(log(2 * x) / 2), 'sol': [Eq(f(x), C1 * sqrt(x) * sin(log(x) / 2) + C2 * sqrt(x) * cos(log(x) / 2) - 2 * sin(log(2 * x) / 2) / 5 - 4 * cos(log(2 * x) / 2) / 5)]}}}