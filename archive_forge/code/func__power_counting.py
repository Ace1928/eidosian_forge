from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants
def _power_counting(num):
    _pow = {0}
    for val in num:
        if val.has(x):
            if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                _pow.add(val.as_base_exp()[1])
            elif val == x:
                _pow.add(val.as_base_exp()[1])
            else:
                _pow.update(_power_counting(val.args))
    return _pow