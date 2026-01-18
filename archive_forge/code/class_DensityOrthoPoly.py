from scipy import stats, integrate, special
import numpy as np
class DensityOrthoPoly:
    """Univariate density estimation by orthonormal series expansion


    Uses an orthonormal polynomial basis to approximate a univariate density.


    currently all arguments can be given to fit, I might change it to requiring
    arguments in __init__ instead.
    """

    def __init__(self, polybase=None, order=5):
        if polybase is not None:
            self.polybase = polybase
            self.polys = polys = [polybase(i) for i in range(order)]
        self._corfactor = 1
        self._corshift = 0

    def fit(self, x, polybase=None, order=5, limits=None):
        """estimate the orthogonal polynomial approximation to the density

        """
        if polybase is None:
            polys = self.polys[:order]
        else:
            self.polybase = polybase
            self.polys = polys = [polybase(i) for i in range(order)]
        if not hasattr(self, 'offsetfac'):
            self.offsetfac = polys[0].offsetfactor
        xmin, xmax = (x.min(), x.max())
        if limits is None:
            self.offset = offset = (xmax - xmin) * self.offsetfac
            limits = self.limits = (xmin - offset, xmax + offset)
        interval_length = limits[1] - limits[0]
        xinterval = xmax - xmin
        self.shrink = 1.0 / interval_length
        offset = (interval_length - xinterval) / 2.0
        self.shift = xmin - offset
        self.x = x = self._transform(x)
        coeffs = [p(x).mean() for p in polys]
        self.coeffs = coeffs
        self.polys = polys
        self._verify()
        return self

    def evaluate(self, xeval, order=None):
        xeval = self._transform(xeval)
        if order is None:
            order = len(self.polys)
        res = sum((c * p(xeval) for c, p in list(zip(self.coeffs, self.polys))[:order]))
        res = self._correction(res)
        return res

    def __call__(self, xeval):
        """alias for evaluate, except no order argument"""
        return self.evaluate(xeval)

    def _verify(self):
        """check for bona fide density correction

        currently only checks that density integrates to 1

`       non-negativity - NotImplementedYet
        """
        intdomain = self.limits
        self._corfactor = 1.0 / integrate.quad(self.evaluate, *intdomain)[0]
        return self._corfactor

    def _correction(self, x):
        """bona fide density correction

        affine shift of density to make it into a proper density

        """
        if self._corfactor != 1:
            x *= self._corfactor
        if self._corshift != 0:
            x += self._corshift
        return x

    def _transform(self, x):
        """transform observation to the domain of the density


        uses shrink and shift attribute which are set in fit to stay


        """
        domain = self.polys[0].domain
        ilen = domain[1] - domain[0]
        shift = self.shift - domain[0] / self.shrink / ilen
        shrink = self.shrink * ilen
        return (x - shift) * shrink