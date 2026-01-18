from sympy.core import Add, Mul, Pow
from sympy.core.numbers import (NaN, Infinity, NegativeInfinity, Float, I, pi,
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Ci, Shi, Si, erf, erfc, erfi)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals import meijerint
from sympy.matrices import Matrix
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains import QQ, RR
from sympy.polys.polyclasses import DMF
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.printing import sstr
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import nsimplify
from sympy.solvers.solvers import solve
from .recurrence import HolonomicSequence, RecurrenceOperator, RecurrenceOperators
from .holonomicerrors import (NotPowerSeriesError, NotHyperSeriesError,
from sympy.integrals.meijerint import _mytype
def _extend_y0(Holonomic, n):
    """
    Tries to find more initial conditions by substituting the initial
    value point in the differential equation.
    """
    if Holonomic.annihilator.is_singular(Holonomic.x0) or Holonomic.is_singularics() == True:
        return Holonomic.y0
    annihilator = Holonomic.annihilator
    a = annihilator.order
    listofpoly = []
    y0 = Holonomic.y0
    R = annihilator.parent.base
    K = R.get_field()
    for i, j in enumerate(annihilator.listofpoly):
        if isinstance(j, annihilator.parent.base.dtype):
            listofpoly.append(K.new(j.rep))
    if len(y0) < a or n <= len(y0):
        return y0
    else:
        list_red = [-listofpoly[i] / listofpoly[a] for i in range(a)]
        if len(y0) > a:
            y1 = [y0[i] for i in range(a)]
        else:
            y1 = list(y0)
        for i in range(n - a):
            sol = 0
            for a, b in zip(y1, list_red):
                r = DMFsubs(b, Holonomic.x0)
                if not getattr(r, 'is_finite', True):
                    return y0
                if isinstance(r, (PolyElement, FracElement)):
                    r = r.as_expr()
                sol += a * r
            y1.append(sol)
            list_red = _derivate_diff_eq(list_red)
        return y0 + y1[len(y0):]