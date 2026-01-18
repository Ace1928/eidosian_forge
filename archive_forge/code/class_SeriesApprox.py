import math
from sympy.sets.sets import Interval
from sympy.calculus.singularities import is_increasing, is_decreasing
from sympy.codegen.rewriting import Optimization
from sympy.core.function import UndefinedFunction
class SeriesApprox(Optimization):
    """ Approximates functions by expanding them as a series.

    Parameters
    ==========

    bounds : dict
        Mapping expressions to length 2 tuple of bounds (low, high).
    reltol : number
        Threshold for when to ignore a term. Taken relative to the largest
        lower bound among bounds.
    max_order : int
        Largest order to include in series expansion
    n_point_checks : int (even)
        The validity of an expansion (with respect to reltol) is checked at
        discrete points (linearly spaced over the bounds of the variable). The
        number of points used in this numerical check is given by this number.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x, y
    >>> from sympy.codegen.rewriting import optimize
    >>> from sympy.codegen.approximations import SeriesApprox
    >>> bounds = {x: (-.1, .1), y: (pi-1, pi+1)}
    >>> series_approx2 = SeriesApprox(bounds, reltol=1e-2)
    >>> series_approx3 = SeriesApprox(bounds, reltol=1e-3)
    >>> series_approx8 = SeriesApprox(bounds, reltol=1e-8)
    >>> expr = sin(x)*sin(y)
    >>> optimize(expr, [series_approx2])
    x*(-y + (y - pi)**3/6 + pi)
    >>> optimize(expr, [series_approx3])
    (-x**3/6 + x)*sin(y)
    >>> optimize(expr, [series_approx8])
    sin(x)*sin(y)

    """

    def __init__(self, bounds, reltol, max_order=4, n_point_checks=4, **kwargs):
        super().__init__(**kwargs)
        self.bounds = bounds
        self.reltol = reltol
        self.max_order = max_order
        if n_point_checks % 2 == 1:
            raise ValueError('Checking the solution at expansion point is not helpful')
        self.n_point_checks = n_point_checks
        self._prec = math.ceil(-math.log10(self.reltol))

    def __call__(self, expr):
        return expr.factor().replace(self.query, lambda arg: self.value(arg))

    def query(self, expr):
        return expr.is_Function and (not isinstance(expr, UndefinedFunction)) and (len(expr.args) == 1)

    def value(self, fexpr):
        free_symbols = fexpr.free_symbols
        if len(free_symbols) != 1:
            return fexpr
        symb, = free_symbols
        if symb not in self.bounds:
            return fexpr
        lo, hi = self.bounds[symb]
        x0 = (lo + hi) / 2
        cheapest = None
        for n in range(self.max_order + 1, 0, -1):
            fseri = fexpr.series(symb, x0=x0, n=n).removeO()
            n_ok = True
            for idx in range(self.n_point_checks):
                x = lo + idx * (hi - lo) / (self.n_point_checks - 1)
                val = fseri.xreplace({symb: x})
                ref = fexpr.xreplace({symb: x})
                if abs((1 - val / ref).evalf(self._prec)) > self.reltol:
                    n_ok = False
                    break
            if n_ok:
                cheapest = fseri
            else:
                break
        if cheapest is None:
            return fexpr
        else:
            return cheapest