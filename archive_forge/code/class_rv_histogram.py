import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
class rv_histogram(rv_continuous):
    """
    Generates a distribution given by a histogram.
    This is useful to generate a template distribution from a binned
    datasample.

    As a subclass of the `rv_continuous` class, `rv_histogram` inherits from it
    a collection of generic methods (see `rv_continuous` for the full list),
    and implements them based on the properties of the provided binned
    datasample.

    Parameters
    ----------
    histogram : tuple of array_like
        Tuple containing two array_like objects.
        The first containing the content of n bins,
        the second containing the (n+1) bin boundaries.
        In particular, the return value of `numpy.histogram` is accepted.

    density : bool, optional
        If False, assumes the histogram is proportional to counts per bin;
        otherwise, assumes it is proportional to a density.
        For constant bin widths, these are equivalent, but the distinction
        is important when bin widths vary (see Notes).
        If None (default), sets ``density=True`` for backwards compatibility,
        but warns if the bin widths are variable. Set `density` explicitly
        to silence the warning.

        .. versionadded:: 1.10.0

    Notes
    -----
    When a histogram has unequal bin widths, there is a distinction between
    histograms that are proportional to counts per bin and histograms that are
    proportional to probability density over a bin. If `numpy.histogram` is
    called with its default ``density=False``, the resulting histogram is the
    number of counts per bin, so ``density=False`` should be passed to
    `rv_histogram`. If `numpy.histogram` is called with ``density=True``, the
    resulting histogram is in terms of probability density, so ``density=True``
    should be passed to `rv_histogram`. To avoid warnings, always pass
    ``density`` explicitly when the input histogram has unequal bin widths.

    There are no additional shape parameters except for the loc and scale.
    The pdf is defined as a stepwise function from the provided histogram.
    The cdf is a linear interpolation of the pdf.

    .. versionadded:: 0.19.0

    Examples
    --------

    Create a scipy.stats distribution from a numpy histogram

    >>> import scipy.stats
    >>> import numpy as np
    >>> data = scipy.stats.norm.rvs(size=100000, loc=0, scale=1.5,
    ...                             random_state=123)
    >>> hist = np.histogram(data, bins=100)
    >>> hist_dist = scipy.stats.rv_histogram(hist, density=False)

    Behaves like an ordinary scipy rv_continuous distribution

    >>> hist_dist.pdf(1.0)
    0.20538577847618705
    >>> hist_dist.cdf(2.0)
    0.90818568543056499

    PDF is zero above (below) the highest (lowest) bin of the histogram,
    defined by the max (min) of the original dataset

    >>> hist_dist.pdf(np.max(data))
    0.0
    >>> hist_dist.cdf(np.max(data))
    1.0
    >>> hist_dist.pdf(np.min(data))
    7.7591907244498314e-05
    >>> hist_dist.cdf(np.min(data))
    0.0

    PDF and CDF follow the histogram

    >>> import matplotlib.pyplot as plt
    >>> X = np.linspace(-5.0, 5.0, 100)
    >>> fig, ax = plt.subplots()
    >>> ax.set_title("PDF from Template")
    >>> ax.hist(data, density=True, bins=100)
    >>> ax.plot(X, hist_dist.pdf(X), label='PDF')
    >>> ax.plot(X, hist_dist.cdf(X), label='CDF')
    >>> ax.legend()
    >>> fig.show()

    """
    _support_mask = rv_continuous._support_mask

    def __init__(self, histogram, *args, density=None, **kwargs):
        """
        Create a new distribution using the given histogram

        Parameters
        ----------
        histogram : tuple of array_like
            Tuple containing two array_like objects.
            The first containing the content of n bins,
            the second containing the (n+1) bin boundaries.
            In particular, the return value of np.histogram is accepted.
        density : bool, optional
            If False, assumes the histogram is proportional to counts per bin;
            otherwise, assumes it is proportional to a density.
            For constant bin widths, these are equivalent.
            If None (default), sets ``density=True`` for backward
            compatibility, but warns if the bin widths are variable. Set
            `density` explicitly to silence the warning.
        """
        self._histogram = histogram
        self._density = density
        if len(histogram) != 2:
            raise ValueError('Expected length 2 for parameter histogram')
        self._hpdf = np.asarray(histogram[0])
        self._hbins = np.asarray(histogram[1])
        if len(self._hpdf) + 1 != len(self._hbins):
            raise ValueError('Number of elements in histogram content and histogram boundaries do not match, expected n and n+1.')
        self._hbin_widths = self._hbins[1:] - self._hbins[:-1]
        bins_vary = not np.allclose(self._hbin_widths, self._hbin_widths[0])
        if density is None and bins_vary:
            message = 'Bin widths are not constant. Assuming `density=True`.Specify `density` explicitly to silence this warning.'
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            density = True
        elif not density:
            self._hpdf = self._hpdf / self._hbin_widths
        self._hpdf = self._hpdf / float(np.sum(self._hpdf * self._hbin_widths))
        self._hcdf = np.cumsum(self._hpdf * self._hbin_widths)
        self._hpdf = np.hstack([0.0, self._hpdf, 0.0])
        self._hcdf = np.hstack([0.0, self._hcdf])
        kwargs['a'] = self.a = self._hbins[0]
        kwargs['b'] = self.b = self._hbins[-1]
        super().__init__(*args, **kwargs)

    def _pdf(self, x):
        """
        PDF of the histogram
        """
        return self._hpdf[np.searchsorted(self._hbins, x, side='right')]

    def _cdf(self, x):
        """
        CDF calculated from the histogram
        """
        return np.interp(x, self._hbins, self._hcdf)

    def _ppf(self, x):
        """
        Percentile function calculated from the histogram
        """
        return np.interp(x, self._hcdf, self._hbins)

    def _munp(self, n):
        """Compute the n-th non-central moment."""
        integrals = (self._hbins[1:] ** (n + 1) - self._hbins[:-1] ** (n + 1)) / (n + 1)
        return np.sum(self._hpdf[1:-1] * integrals)

    def _entropy(self):
        """Compute entropy of distribution"""
        res = _lazywhere(self._hpdf[1:-1] > 0.0, (self._hpdf[1:-1],), np.log, 0.0)
        return -np.sum(self._hpdf[1:-1] * res * self._hbin_widths)

    def _updated_ctor_param(self):
        """
        Set the histogram as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        dct['histogram'] = self._histogram
        dct['density'] = self._density
        return dct