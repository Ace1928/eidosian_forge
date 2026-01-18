from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import (Function, Derivative, AppliedUndef, diff,
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import (BooleanAtom, BooleanTrue,
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import (Poly, terms_gcd, PolynomialError, lcm)
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import (collect, logcombine, powsimp,  # type: ignore
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve
from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from .single import SingleODEProblem, SingleODESolver, solver_map
def _frobenius(n, m, p0, q0, p, q, x0, x, c, check=None):
    """
    Returns a dict with keys as coefficients and values as their values in terms of C0
    """
    n = int(n)
    m2 = check
    d = Dummy('d')
    numsyms = numbered_symbols('C', start=0)
    numsyms = [next(numsyms) for i in range(n + 1)]
    serlist = []
    for ser in [p, q]:
        if ser.is_polynomial(x) and Poly(ser, x).degree() <= n:
            if x0:
                ser = ser.subs(x, x + x0)
            dict_ = Poly(ser, x).as_dict()
        else:
            tseries = series(ser, x=x0, n=n + 1)
            dict_ = Poly(list(ordered(tseries.args))[:-1], x).as_dict()
        for i in range(n + 1):
            if (i,) not in dict_:
                dict_[i,] = S.Zero
        serlist.append(dict_)
    pseries = serlist[0]
    qseries = serlist[1]
    indicial = d * (d - 1) + d * p0 + q0
    frobdict = {}
    for i in range(1, n + 1):
        num = c * (m * pseries[i,] + qseries[i,])
        for j in range(1, i):
            sym = Symbol('C' + str(j))
            num += frobdict[sym] * ((m + j) * pseries[i - j,] + qseries[i - j,])
        if m2 is not None and i == m2 - m:
            if num:
                return False
            else:
                frobdict[numsyms[i]] = S.Zero
        else:
            frobdict[numsyms[i]] = -num / indicial.subs(d, m + i)
    return frobdict