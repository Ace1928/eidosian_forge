from collections import defaultdict
from sympy.core.numbers import (nan, oo, zoo)
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable
def _compute_fps(f, x, x0, dir, hyper, order, rational, full):
    """Recursive wrapper to compute fps.

    See :func:`compute_fps` for details.
    """
    if x0 in [S.Infinity, S.NegativeInfinity]:
        dir = S.One if x0 is S.Infinity else -S.One
        temp = f.subs(x, 1 / x)
        result = _compute_fps(temp, x, 0, dir, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, 1 / x), result[2].subs(x, 1 / x))
    elif x0 or dir == -S.One:
        if dir == -S.One:
            rep = -x + x0
            rep2 = -x
            rep2b = x0
        else:
            rep = x + x0
            rep2 = x
            rep2b = -x0
        temp = f.subs(x, rep)
        result = _compute_fps(temp, x, 0, S.One, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, rep2 + rep2b), result[2].subs(x, rep2 + rep2b))
    if f.is_polynomial(x):
        k = Dummy('k')
        ak = sequence(Coeff(f, x, k), (k, 1, oo))
        xk = sequence(x ** k, (k, 0, oo))
        ind = f.coeff(x, 0)
        return (ak, xk, ind)
    if isinstance(f, Add):
        result = False
        ak = sequence(S.Zero, (0, oo))
        ind, xk = (S.Zero, None)
        for t in Add.make_args(f):
            res = _compute_fps(t, x, 0, S.One, hyper, order, rational, full)
            if res:
                if not result:
                    result = True
                    xk = res[1]
                if res[0].start > ak.start:
                    seq = ak
                    s, f = (ak.start, res[0].start)
                else:
                    seq = res[0]
                    s, f = (res[0].start, ak.start)
                save = Add(*[z[0] * z[1] for z in zip(seq[0:f - s], xk[s:f])])
                ak += res[0]
                ind += res[2] + save
            else:
                ind += t
        if result:
            return (ak, xk, ind)
        return None
    syms = f.free_symbols.difference({x})
    f, symb = expand(f).as_independent(*syms)
    result = None
    k = Dummy('k')
    if rational:
        result = rational_algorithm(f, x, k, order, full)
    if result is None and hyper:
        result = hyper_algorithm(f, x, k, order)
    if result is None:
        return None
    from sympy.simplify.powsimp import powsimp
    if symb.is_zero:
        symb = S.One
    else:
        symb = powsimp(symb)
    ak = sequence(result[0], (k, result[2], oo))
    xk_formula = powsimp(x ** k * symb)
    xk = sequence(xk_formula, (k, 0, oo))
    ind = powsimp(result[1] * symb)
    return (ak, xk, ind)