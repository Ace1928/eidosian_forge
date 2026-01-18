from functools import reduce, wraps
from itertools import repeat
from sympy.core import S, pi
from sympy.core.add import Add
from sympy.core.function import (
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.core.traversal import postorder_traversal
from sympy.functions.combinatorial.factorials import factorial, rf
from sympy.functions.elementary.complexes import re, arg, Abs
from sympy.functions.elementary.exponential import exp, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.functions.elementary.trigonometric import cos, cot, sin, tan
from sympy.functions.special.bessel import besselj
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import meijerg
from sympy.integrals import integrate, Integral
from sympy.integrals.meijerint import _dummy
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.polys.polyroots import roots
from sympy.polys.polytools import factor, Poly
from sympy.polys.rootoftools import CRootOf
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
import sympy.integrals.laplace as _laplace
def _noconds_(default):
    """
    This is a decorator generator for dropping convergence conditions.

    Explanation
    ===========

    Suppose you define a function ``transform(*args)`` which returns a tuple of
    the form ``(result, cond1, cond2, ...)``.

    Decorating it ``@_noconds_(default)`` will add a new keyword argument
    ``noconds`` to it. If ``noconds=True``, the return value will be altered to
    be only ``result``, whereas if ``noconds=False`` the return value will not
    be altered.

    The default value of the ``noconds`` keyword will be ``default`` (i.e. the
    argument of this function).
    """

    def make_wrapper(func):

        @wraps(func)
        def wrapper(*args, noconds=default, **kwargs):
            res = func(*args, **kwargs)
            if noconds:
                return res[0]
            return res
        return wrapper
    return make_wrapper