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
def _solve_explike_DE(f, x, DE, g, k):
    """Solves DE with constant coefficients."""
    from sympy.solvers import rsolve
    for t in Add.make_args(DE):
        coeff, d = t.as_independent(g)
        if coeff.free_symbols:
            return
    RE = exp_re(DE, g, k)
    init = {}
    for i in range(len(Add.make_args(RE))):
        if i:
            f = f.diff(x)
        init[g(k).subs(k, i)] = f.limit(x, 0)
    sol = rsolve(RE, g(k), init)
    if sol:
        return (sol / factorial(k), S.Zero, S.Zero)