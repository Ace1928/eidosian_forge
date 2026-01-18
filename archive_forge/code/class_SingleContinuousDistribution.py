from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda, PoleError
from sympy.core.numbers import (I, nan, oo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import poly
from sympy.series.series import series
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Union)
from sympy.solvers.solveset import solveset
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy.stats.rv import (RandomDomain, SingleDomain, ConditionalDomain, is_random,
class SingleContinuousDistribution(ContinuousDistribution, NamedArgsMixin):
    """ Continuous distribution of a single variable.

    Explanation
    ===========

    Serves as superclass for Normal/Exponential/UniformDistribution etc....

    Represented by parameters for each of the specific classes.  E.g
    NormalDistribution is represented by a mean and standard deviation.

    Provides methods for pdf, cdf, and sampling.

    See Also
    ========

    sympy.stats.crv_types.*
    """
    set = Interval(-oo, oo)

    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    @cacheit
    def compute_cdf(self, **kwargs):
        """ Compute the CDF from the PDF.

        Returns a Lambda.
        """
        x, z = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.set.start
        pdf = self.pdf(x)
        cdf = integrate(pdf.doit(), (x, left_bound, z), **kwargs)
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    def _cdf(self, x):
        return None

    def cdf(self, x, **kwargs):
        """ Cumulative density function """
        if len(kwargs) == 0:
            cdf = self._cdf(x)
            if cdf is not None:
                return cdf
        return self.compute_cdf(**kwargs)(x)

    @cacheit
    def compute_characteristic_function(self, **kwargs):
        """ Compute the characteristic function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        cf = integrate(exp(I * t * x) * pdf, (x, self.set))
        return Lambda(t, cf)

    def _characteristic_function(self, t):
        return None

    def characteristic_function(self, t, **kwargs):
        """ Characteristic function """
        if len(kwargs) == 0:
            cf = self._characteristic_function(t)
            if cf is not None:
                return cf
        return self.compute_characteristic_function(**kwargs)(t)

    @cacheit
    def compute_moment_generating_function(self, **kwargs):
        """ Compute the moment generating function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        mgf = integrate(exp(t * x) * pdf, (x, self.set))
        return Lambda(t, mgf)

    def _moment_generating_function(self, t):
        return None

    def moment_generating_function(self, t, **kwargs):
        """ Moment generating function """
        if not kwargs:
            mgf = self._moment_generating_function(t)
            if mgf is not None:
                return mgf
        return self.compute_moment_generating_function(**kwargs)(t)

    def expectation(self, expr, var, evaluate=True, **kwargs):
        """ Expectation of expression over distribution """
        if evaluate:
            try:
                p = poly(expr, var)
                if p.is_zero:
                    return S.Zero
                t = Dummy('t', real=True)
                mgf = self._moment_generating_function(t)
                if mgf is None:
                    return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
                deg = p.degree()
                taylor = poly(series(mgf, t, 0, deg + 1).removeO(), t)
                result = 0
                for k in range(deg + 1):
                    result += p.coeff_monomial(var ** k) * taylor.coeff_monomial(t ** k) * factorial(k)
                return result
            except PolynomialError:
                return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
        else:
            return Integral(expr * self.pdf(var), (var, self.set), **kwargs)

    @cacheit
    def compute_quantile(self, **kwargs):
        """ Compute the Quantile from the PDF.

        Returns a Lambda.
        """
        x, p = symbols('x, p', real=True, cls=Dummy)
        left_bound = self.set.start
        pdf = self.pdf(x)
        cdf = integrate(pdf, (x, left_bound, x), **kwargs)
        quantile = solveset(cdf - p, x, self.set)
        return Lambda(p, Piecewise((quantile, (p >= 0) & (p <= 1)), (nan, True)))

    def _quantile(self, x):
        return None

    def quantile(self, x, **kwargs):
        """ Cumulative density function """
        if len(kwargs) == 0:
            quantile = self._quantile(x)
            if quantile is not None:
                return quantile
        return self.compute_quantile(**kwargs)(x)