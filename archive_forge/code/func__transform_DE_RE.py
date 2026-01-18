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
def _transform_DE_RE(DE, g, k, order, syms):
    """Converts DE with free parameters into RE of hypergeometric type."""
    from sympy.solvers.solveset import linsolve
    RE = hyper_re(DE, g, k)
    eq = []
    for i in range(1, order):
        coeff = RE.coeff(g(k + i))
        eq.append(coeff)
    sol = dict(zip(syms, (i for s in linsolve(eq, list(syms)) for i in s)))
    if sol:
        m = Wild('m')
        RE = RE.subs(sol)
        RE = RE.factor().as_numer_denom()[0].collect(g(k + m))
        RE = RE.as_coeff_mul(g)[1][0]
        for i in range(order):
            if RE.coeff(g(k + i)) and i:
                RE = RE.subs(k, k - i)
                break
    return RE