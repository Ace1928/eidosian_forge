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
def _convert_meijerint(func, x, initcond=True, domain=QQ):
    args = meijerint._rewrite1(func, x)
    if args:
        fac, po, g, _ = args
    else:
        return None
    fac_list = [fac * i[0] for i in g]
    t = po.as_base_exp()
    s = t[1] if t[0] == x else S.Zero
    po_list = [s + i[1] for i in g]
    G_list = [i[2] for i in g]

    def _shift(func, s):
        z = func.args[-1]
        if z.has(I):
            z = z.subs(exp_polar, exp)
        d = z.collect(x, evaluate=False)
        b = list(d)[0]
        a = d[b]
        t = b.as_base_exp()
        b = t[1] if t[0] == x else S.Zero
        r = s / b
        an = (i + r for i in func.args[0][0])
        ap = (i + r for i in func.args[0][1])
        bm = (i + r for i in func.args[1][0])
        bq = (i + r for i in func.args[1][1])
        return (a ** (-r), meijerg((an, ap), (bm, bq), z))
    coeff, m = _shift(G_list[0], po_list[0])
    sol = fac_list[0] * coeff * from_meijerg(m, initcond=initcond, domain=domain)
    for i in range(1, len(G_list)):
        coeff, m = _shift(G_list[i], po_list[i])
        sol += fac_list[i] * coeff * from_meijerg(m, initcond=initcond, domain=domain)
    return sol