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
class FormalPowerSeriesProduct(FiniteFormalPowerSeries):
    """Represents the product of two formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesProduct` object. The first argument contains the two
    functions involved in the product. Also, the coefficient sequence contains
    both the coefficient sequence of the formal power series of the involved functions.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    def __init__(self, *args):
        ffps, gfps = (self.ffps, self.gfps)
        k = ffps.ak.variables[0]
        self.coeff1 = sequence(ffps.ak.formula, (k, 0, oo))
        k = gfps.ak.variables[0]
        self.coeff2 = sequence(gfps.ak.formula, (k, 0, oo))

    @property
    def function(self):
        """Function of the product of two formal power series."""
        return self.f * self.g

    def _eval_terms(self, n):
        """
        Returns the first ``n`` terms of the product formal power series.
        Term by term logic is implemented here.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))
        >>> fprod = f1.product(f2, x)

        >>> fprod._eval_terms(4)
        x**3/3 + x**2 + x

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.product

        """
        coeff1, coeff2 = (self.coeff1, self.coeff2)
        aks = convolution(coeff1[:n], coeff2[:n])
        terms = []
        for i in range(0, n):
            terms.append(aks[i] * self.ffps.xk.coeff(i))
        return Add(*terms)