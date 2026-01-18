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
def _get_examples_ode_sol_2nd_2F1_hypergeometric():
    return {'hint': '2nd_hypergeometric', 'func': f(x), 'examples': {'2nd_2F1_hyper_01': {'eq': x * (x - 1) * f(x).diff(x, 2) + (S(3) / 2 - 2 * x) * f(x).diff(x) + 2 * f(x), 'sol': [Eq(f(x), C1 * x ** (S(5) / 2) * hyper((S(3) / 2, S(1) / 2), (S(7) / 2,), x) + C2 * hyper((-1, -2), (-S(3) / 2,), x))]}, '2nd_2F1_hyper_02': {'eq': x * (x - 1) * f(x).diff(x, 2) + S(7) / 2 * x * f(x).diff(x) + f(x), 'sol': [Eq(f(x), (C1 * (1 - x) ** (S(5) / 2) * hyper((S(1) / 2, 2), (S(7) / 2,), 1 - x) + C2 * hyper((-S(1) / 2, -2), (-S(3) / 2,), 1 - x)) / (x - 1) ** (S(5) / 2))]}, '2nd_2F1_hyper_03': {'eq': x * (x - 1) * f(x).diff(x, 2) + (S(3) + S(7) / 2 * x) * f(x).diff(x) + f(x), 'sol': [Eq(f(x), (C1 * (1 - x) ** (S(11) / 2) * hyper((S(1) / 2, 2), (S(13) / 2,), 1 - x) + C2 * hyper((-S(7) / 2, -5), (-S(9) / 2,), 1 - x)) / (x - 1) ** (S(11) / 2))]}, '2nd_2F1_hyper_04': {'eq': -x ** (S(5) / 7) * (-416 * x ** (S(9) / 7) / 9 - 2385 * x ** (S(5) / 7) / 49 + S(298) * x / 3) * f(x) / (196 * (-x ** (S(6) / 7) + x) ** 2 * (x ** (S(6) / 7) + x) ** 2) + Derivative(f(x), (x, 2)), 'sol': [Eq(f(x), x ** (S(45) / 98) * (C1 * x ** (S(4) / 49) * hyper((S(1) / 3, -S(1) / 2), (S(9) / 7,), x ** (S(2) / 7)) + C2 * hyper((S(1) / 21, -S(11) / 14), (S(5) / 7,), x ** (S(2) / 7))) / (x ** (S(2) / 7) - 1) ** (S(19) / 84))], 'checkodesol_XFAIL': True}}}