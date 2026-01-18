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
def _add_example_keys(func):

    def inner():
        solver = func()
        examples = []
        for example in solver['examples']:
            temp = {'eq': solver['examples'][example]['eq'], 'sol': solver['examples'][example]['sol'], 'XFAIL': solver['examples'][example].get('XFAIL', []), 'func': solver['examples'][example].get('func', solver['func']), 'example_name': example, 'slow': solver['examples'][example].get('slow', False), 'simplify_flag': solver['examples'][example].get('simplify_flag', True), 'checkodesol_XFAIL': solver['examples'][example].get('checkodesol_XFAIL', False), 'dsolve_too_slow': solver['examples'][example].get('dsolve_too_slow', False), 'checkodesol_too_slow': solver['examples'][example].get('checkodesol_too_slow', False), 'hint': solver['hint']}
            examples.append(temp)
        return examples
    return inner()