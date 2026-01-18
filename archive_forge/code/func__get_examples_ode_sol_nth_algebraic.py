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
def _get_examples_ode_sol_nth_algebraic():
    M, m, r, t = symbols('M m r t')
    phi = Function('phi')
    k = Symbol('k')
    return {'hint': 'nth_algebraic', 'func': f(x), 'examples': {'algeb_01': {'eq': f(x) * f(x).diff(x) * f(x).diff(x, x) * (f(x) - 1) * (f(x).diff(x) - x), 'sol': [Eq(f(x), C1 + x ** 2 / 2), Eq(f(x), C1 + C2 * x)]}, 'algeb_02': {'eq': f(x) * f(x).diff(x) * f(x).diff(x, x) * (f(x) - 1), 'sol': [Eq(f(x), C1 + C2 * x)]}, 'algeb_03': {'eq': f(x) * f(x).diff(x) * f(x).diff(x, x), 'sol': [Eq(f(x), C1 + C2 * x)]}, 'algeb_04': {'eq': Eq(-M * phi(t).diff(t), Rational(3, 2) * m * r ** 2 * phi(t).diff(t) * phi(t).diff(t, t)), 'sol': [Eq(phi(t), C1), Eq(phi(t), C1 + C2 * t - M * t ** 2 / (3 * m * r ** 2))], 'func': phi(t)}, 'algeb_05': {'eq': (1 - sin(f(x))) * f(x).diff(x), 'sol': [Eq(f(x), C1)], 'XFAIL': ['separable']}, 'algeb_06': {'eq': (diff(f(x)) - x) * (diff(f(x)) + x), 'sol': [Eq(f(x), C1 - x ** 2 / 2), Eq(f(x), C1 + x ** 2 / 2)]}, 'algeb_07': {'eq': Eq(Derivative(f(x), x), Derivative(g(x), x)), 'sol': [Eq(f(x), C1 + g(x))]}, 'algeb_08': {'eq': f(x).diff(x) - C1, 'sol': [Eq(f(x), C1 * x + C2)]}, 'algeb_09': {'eq': f(x) * f(x).diff(x), 'sol': [Eq(f(x), C1)]}, 'algeb_10': {'eq': (diff(f(x)) - x) * (diff(f(x)) + x), 'sol': [Eq(f(x), C1 - x ** 2 / 2), Eq(f(x), C1 + x ** 2 / 2)]}, 'algeb_11': {'eq': f(x) + f(x) * f(x).diff(x), 'sol': [Eq(f(x), 0), Eq(f(x), C1 - x)], 'XFAIL': ['separable', '1st_exact', '1st_linear', 'Bernoulli', '1st_homogeneous_coeff_best', '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_dep_div_indep', 'lie_group', 'nth_linear_constant_coeff_undetermined_coefficients', 'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients', 'nth_linear_constant_coeff_variation_of_parameters', 'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters']}, 'algeb_12': {'eq': Derivative(x * f(x), x, x, x), 'sol': [Eq(f(x), (C1 + C2 * x + C3 * x ** 2) / x)], 'XFAIL': ['nth_algebraic']}, 'algeb_13': {'eq': Eq(Derivative(x * Derivative(f(x), x), x) / x, exp(x)), 'sol': [Eq(f(x), C1 + C2 * log(x) + exp(x) - Ei(x))], 'XFAIL': ['nth_algebraic']}, 'algeb_14': {'eq': Eq(f(x).diff(x), 0), 'sol': [Eq(f(x), C1)]}, 'algeb_15': {'eq': Eq(3 * f(x).diff(x) - 5, 0), 'sol': [Eq(f(x), C1 + x * Rational(5, 3))]}, 'algeb_16': {'eq': Eq(3 * f(x).diff(x), 5), 'sol': [Eq(f(x), C1 + x * Rational(5, 3))]}, 'algeb_17': {'eq': Eq(3 * f(x).diff(x) - 1, 0), 'sol': [Eq(f(x), C1 + x / 3)]}, 'algeb_18': {'eq': Eq(x * f(x).diff(x) - 1, 0), 'sol': [Eq(f(x), C1 + log(x))]}, 'algeb_19': {'eq': f(x).diff(x) - x * exp(-k * x), 'sol': [Eq(f(x), C1 + Piecewise(((-k * x - 1) * exp(-k * x) / k ** 2, Ne(k ** 2, 0)), (x ** 2 / 2, True)))]}, 'algeb_20': {'eq': -f(x).diff(x) + x * exp(-k * x), 'sol': [Eq(f(x), C1 + Piecewise(((-k * x - 1) * exp(-k * x) / k ** 2, Ne(k ** 2, 0)), (x ** 2 / 2, True)))]}, 'algeb_21': {'eq': Eq(g(x).diff(x).diff(x), (x - 2) ** 2 + (x - 3) ** 3), 'sol': [Eq(g(x), C1 + C2 * x + x ** 5 / 20 - 2 * x ** 4 / 3 + 23 * x ** 3 / 6 - 23 * x ** 2 / 2)], 'func': g(x)}, 'algeb_22': {'eq': f(x).diff(x) - C1 * g(x).diff(x), 'sol': [Eq(f(x), C2 + C1 * g(x))], 'func': f(x)}, 'algeb_23': {'eq': f(x).diff(x) - 3 * C1 - 3 * x ** 2, 'sol': [Eq(f(x), C2 + 3 * C1 * x + x ** 3)]}}}