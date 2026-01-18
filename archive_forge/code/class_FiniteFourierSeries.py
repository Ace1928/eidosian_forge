from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence
class FiniteFourierSeries(FourierSeries):
    """Represents Finite Fourier sine/cosine series.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    Parameters
    ==========

    f : Expr
        Expression for finding fourier_series

    limits : ( x, start, stop)
        x is the independent variable for the expression f
        (start, stop) is the period of the fourier series

    exprs: (a0, an, bn) or Expr
        a0 is the constant term a0 of the fourier series
        an is a dictionary of coefficients of cos terms
         an[k] = coefficient of cos(pi*(k/L)*x)
        bn is a dictionary of coefficients of sin terms
         bn[k] = coefficient of sin(pi*(k/L)*x)

        or exprs can be an expression to be converted to fourier form

    Methods
    =======

    This class is an extension of FourierSeries class.
    Please refer to sympy.series.fourier.FourierSeries for
    further information.

    See Also
    ========

    sympy.series.fourier.FourierSeries
    sympy.series.fourier.fourier_series
    """

    def __new__(cls, f, limits, exprs):
        f = sympify(f)
        limits = sympify(limits)
        exprs = sympify(exprs)
        if not (isinstance(exprs, Tuple) and len(exprs) == 3):
            c, e = exprs.as_coeff_add()
            from sympy.simplify.fu import TR10
            rexpr = c + Add(*[TR10(i) for i in e])
            a0, exp_ls = rexpr.expand(trig=False, power_base=False, power_exp=False, log=False).as_coeff_add()
            x = limits[0]
            L = abs(limits[2] - limits[1]) / 2
            a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k is not S.Zero])
            b = Wild('b', properties=[lambda k: x not in k.free_symbols])
            an = {}
            bn = {}
            for p in exp_ls:
                t = p.match(b * cos(a * (pi / L) * x))
                q = p.match(b * sin(a * (pi / L) * x))
                if t:
                    an[t[a]] = t[b] + an.get(t[a], S.Zero)
                elif q:
                    bn[q[a]] = q[b] + bn.get(q[a], S.Zero)
                else:
                    a0 += p
            exprs = Tuple(a0, an, bn)
        return Expr.__new__(cls, f, limits, exprs)

    @property
    def interval(self):
        _length = 1 if self.a0 else 0
        _length += max(set(self.an.keys()).union(set(self.bn.keys()))) + 1
        return Interval(0, _length)

    @property
    def length(self):
        return self.stop - self.start

    def shiftx(self, s):
        s, x = (sympify(s), self.x)
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        _expr = self.truncate().subs(x, x + s)
        sfunc = self.function.subs(x, x + s)
        return self.func(sfunc, self.args[1], _expr)

    def scale(self, s):
        s, x = (sympify(s), self.x)
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        _expr = self.truncate() * s
        sfunc = self.function * s
        return self.func(sfunc, self.args[1], _expr)

    def scalex(self, s):
        s, x = (sympify(s), self.x)
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        _expr = self.truncate().subs(x, x * s)
        sfunc = self.function.subs(x, x * s)
        return self.func(sfunc, self.args[1], _expr)

    def _eval_term(self, pt):
        if pt == 0:
            return self.a0
        _term = self.an.get(pt, S.Zero) * cos(pt * (pi / self.L) * self.x) + self.bn.get(pt, S.Zero) * sin(pt * (pi / self.L) * self.x)
        return _term

    def __add__(self, other):
        if isinstance(other, FourierSeries):
            return other.__add__(fourier_series(self.function, self.args[1], finite=False))
        elif isinstance(other, FiniteFourierSeries):
            if self.period != other.period:
                raise ValueError('Both the series should have same periods')
            x, y = (self.x, other.x)
            function = self.function + other.function.subs(y, x)
            if self.x not in function.free_symbols:
                return function
            return fourier_series(function, limits=self.args[1])