from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
class rv_discrete(rv_generic):
    """A generic discrete random variable class meant for subclassing.

    `rv_discrete` is a base class to construct specific distribution classes
    and instances for discrete random variables. It can also be used
    to construct an arbitrary distribution defined by a list of support
    points and corresponding probabilities.

    Parameters
    ----------
    a : float, optional
        Lower bound of the support of the distribution, default: 0
    b : float, optional
        Upper bound of the support of the distribution, default: plus infinity
    moment_tol : float, optional
        The tolerance for the generic calculation of moments.
    values : tuple of two array_like, optional
        ``(xk, pk)`` where ``xk`` are integers and ``pk`` are the non-zero
        probabilities between 0 and 1 with ``sum(pk) = 1``. ``xk``
        and ``pk`` must have the same shape, and ``xk`` must be unique.
    inc : integer, optional
        Increment for the support of the distribution.
        Default is 1. (other values have not been tested)
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example "m, n" for a distribution
        that takes two integers as the two shape arguments for all its methods
        If not provided, shape parameters will be inferred from
        the signatures of the private methods, ``_pmf`` and ``_cdf`` of
        the instance.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
    pmf
    logpmf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    support

    Notes
    -----
    This class is similar to `rv_continuous`. Whether a shape parameter is
    valid is decided by an ``_argcheck`` method (which defaults to checking
    that its arguments are strictly positive.)
    The main differences are as follows.

    - The support of the distribution is a set of integers.
    - Instead of the probability density function, ``pdf`` (and the
      corresponding private ``_pdf``), this class defines the
      *probability mass function*, `pmf` (and the corresponding
      private ``_pmf``.)
    - There is no ``scale`` parameter.
    - The default implementations of methods (e.g. ``_cdf``) are not designed
      for distributions with support that is unbounded below (i.e.
      ``a=-np.inf``), so they must be overridden.

    To create a new discrete distribution, we would do the following:

    >>> from scipy.stats import rv_discrete
    >>> class poisson_gen(rv_discrete):
    ...     "Poisson distribution"
    ...     def _pmf(self, k, mu):
    ...         return exp(-mu) * mu**k / factorial(k)

    and create an instance::

    >>> poisson = poisson_gen(name="poisson")

    Note that above we defined the Poisson distribution in the standard form.
    Shifting the distribution can be done by providing the ``loc`` parameter
    to the methods of the instance. For example, ``poisson.pmf(x, mu, loc)``
    delegates the work to ``poisson._pmf(x-loc, mu)``.

    **Discrete distributions from a list of probabilities**

    Alternatively, you can construct an arbitrary discrete rv defined
    on a finite set of values ``xk`` with ``Prob{X=xk} = pk`` by using the
    ``values`` keyword argument to the `rv_discrete` constructor.

    **Deepcopying / Pickling**

    If a distribution or frozen distribution is deepcopied (pickled/unpickled,
    etc.), any underlying random number generator is deepcopied with it. An
    implication is that if a distribution relies on the singleton RandomState
    before copying, it will rely on a copy of that random state after copying,
    and ``np.random.seed`` will no longer control the state.

    Examples
    --------
    Custom made discrete distribution:

    >>> import numpy as np
    >>> from scipy import stats
    >>> xk = np.arange(7)
    >>> pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
    >>> custm = stats.rv_discrete(name='custm', values=(xk, pk))
    >>>
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
    >>> ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
    >>> plt.show()

    Random number generation:

    >>> R = custm.rvs(size=100)

    """

    def __new__(cls, a=0, b=inf, name=None, badvalue=None, moment_tol=1e-08, values=None, inc=1, longname=None, shapes=None, seed=None):
        if values is not None:
            return super().__new__(rv_sample)
        else:
            return super().__new__(cls)

    def __init__(self, a=0, b=inf, name=None, badvalue=None, moment_tol=1e-08, values=None, inc=1, longname=None, shapes=None, seed=None):
        super().__init__(seed)
        self._ctor_param = dict(a=a, b=b, name=name, badvalue=badvalue, moment_tol=moment_tol, values=values, inc=inc, longname=longname, shapes=shapes, seed=seed)
        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.a = a
        self.b = b
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        if values is not None:
            raise ValueError('rv_discrete.__init__(..., values != None, ...)')
        self._construct_argparser(meths_to_inspect=[self._pmf, self._cdf], locscale_in='loc=0', locscale_out='loc, 1')
        self._attach_methods()
        self._construct_docstrings(name, longname)

    def __getstate__(self):
        dct = self.__dict__.copy()
        attrs = ['_parse_args', '_parse_args_stats', '_parse_args_rvs', '_cdfvec', '_ppfvec', 'generic_moment']
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        """Attaches dynamically created methods to the rv_discrete instance."""
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self.vecentropy = vectorize(self._entropy)
        self._attach_argparser_methods()
        _vec_generic_moment = vectorize(_drv2_moment, otypes='d')
        _vec_generic_moment.nin = self.numargs + 2
        self.generic_moment = types.MethodType(_vec_generic_moment, self)
        _vppf = vectorize(_drv2_ppfsingle, otypes='d')
        _vppf.nin = self.numargs + 2
        self._ppfvec = types.MethodType(_vppf, self)
        self._cdfvec.nin = self.numargs + 1

    def _construct_docstrings(self, name, longname):
        if name is None:
            name = 'Distribution'
        self.name = name
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = 'An '
            else:
                hstr = 'A '
            longname = hstr + name
        if sys.flags.optimize < 2:
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname, docdict=docdict_discrete, discrete='discrete')
            else:
                dct = dict(distdiscrete)
                self._construct_doc(docdict_discrete, dct.get(self.name))
            self.__doc__ = self.__doc__.replace('\n    scale : array_like, optional\n        scale parameter (default=1)', '')

    def _updated_ctor_param(self):
        """Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['badvalue'] = self.badvalue
        dct['moment_tol'] = self.moment_tol
        dct['inc'] = self.inc
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    def _nonzero(self, k, *args):
        return floor(k) == k

    def _pmf(self, k, *args):
        return self._cdf(k, *args) - self._cdf(k - 1, *args)

    def _logpmf(self, k, *args):
        return log(self._pmf(k, *args))

    def _logpxf(self, k, *args):
        return self._logpmf(k, *args)

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-1]
            scale = 1
            args = tuple(theta[:-1])
        except IndexError as e:
            raise ValueError('Not enough input arguments.') from e
        return (loc, scale, args)

    def _cdf_single(self, k, *args):
        _a, _b = self._get_support(*args)
        m = arange(int(_a), k + 1)
        return np.sum(self._pmf(m, *args), axis=0)

    def _cdf(self, x, *args):
        k = floor(x)
        return self._cdfvec(k, *args)

    def rvs(self, *args, **kwargs):
        """Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        size : int or tuple of ints, optional
            Defining number of random variates (Default is 1). Note that `size`
            has to be given as keyword, not as positional argument.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `random_state` is None (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance, that instance is used.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        kwargs['discrete'] = True
        return super().rvs(*args, **kwargs)

    def pmf(self, k, *args, **kwds):
        """Probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        pmf : array_like
            Probability mass function evaluated at k

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b)
        if not isinstance(self, rv_sample):
            cond1 = cond1 & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logpmf(self, k, *args, **kwds):
        """Log of the probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter. Default is 0.

        Returns
        -------
        logpmf : array_like
            Log of the probability mass function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b)
        if not isinstance(self, rv_sample):
            cond1 = cond1 & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, self._logpmf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, k, *args, **kwds):
        """Cumulative distribution function of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = k >= _b
        cond3 = np.isneginf(k)
        cond = cond0 & cond1 & np.isfinite(k)
        output = zeros(shape(cond), 'd')
        place(output, cond2 * (cond0 == cond0), 1.0)
        place(output, cond3 * (cond0 == cond0), 0.0)
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, np.clip(self._cdf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, k, *args, **kwds):
        """Log of the cumulative distribution function at k of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = k >= _b
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        place(output, cond2 * (cond0 == cond0), 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, k, *args, **kwds):
        """Survival function (1 - `cdf`) at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        sf : array_like
            Survival function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = ((k < _a) | np.isneginf(k)) & cond0
        cond = cond0 & cond1 & np.isfinite(k)
        output = zeros(shape(cond), 'd')
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, np.clip(self._sf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, k, *args, **kwds):
        """Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as 1 - `cdf`,
        evaluated at `k`.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k < _a) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Lower tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : array_like
            Quantile corresponding to the lower tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        place(output, (q == 0) * (cond == cond), _a - 1 + loc)
        place(output, cond2, _b + loc)
        if np.any(cond):
            goodargs = argsreduce(cond, *(q,) + args + (loc,))
            loc, goodargs = (goodargs[-1], goodargs[:-1])
            place(output, cond, self._ppf(*goodargs) + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Upper tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : ndarray or scalar
            Quantile corresponding to the upper tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond3 = (q == 0) & cond0
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        lower_bound = _a - 1 + loc
        upper_bound = _b + loc
        place(output, cond2 * (cond == cond), lower_bound)
        place(output, cond3 * (cond == cond), upper_bound)
        if np.any(cond):
            goodargs = argsreduce(cond, *(q,) + args + (loc,))
            loc, goodargs = (goodargs[-1], goodargs[:-1])
            place(output, cond, self._isf(*goodargs) + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _entropy(self, *args):
        if hasattr(self, 'pk'):
            return stats.entropy(self.pk)
        else:
            _a, _b = self._get_support(*args)
            return _expect(lambda x: entr(self.pmf(x, *args)), _a, _b, self.ppf(0.5, *args), self.inc)

    def expect(self, func=None, args=(), loc=0, lb=None, ub=None, conditional=False, maxcount=1000, tolerance=1e-10, chunksize=32):
        """
        Calculate expected value of a function with respect to the distribution
        for discrete distribution by numerical summation.

        Parameters
        ----------
        func : callable, optional
            Function for which the expectation value is calculated.
            Takes only one argument.
            The default is the identity mapping f(k) = k.
        args : tuple, optional
            Shape parameters of the distribution.
        loc : float, optional
            Location parameter.
            Default is 0.
        lb, ub : int, optional
            Lower and upper bound for the summation, default is set to the
            support of the distribution, inclusive (``lb <= k <= ub``).
        conditional : bool, optional
            If true then the expectation is corrected by the conditional
            probability of the summation interval. The return value is the
            expectation of the function, `func`, conditional on being in
            the given interval (k such that ``lb <= k <= ub``).
            Default is False.
        maxcount : int, optional
            Maximal number of terms to evaluate (to avoid an endless loop for
            an infinite sum). Default is 1000.
        tolerance : float, optional
            Absolute tolerance for the summation. Default is 1e-10.
        chunksize : int, optional
            Iterate over the support of a distributions in chunks of this size.
            Default is 32.

        Returns
        -------
        expect : float
            Expected value.

        Notes
        -----
        For heavy-tailed distributions, the expected value may or
        may not exist,
        depending on the function, `func`. If it does exist, but the
        sum converges
        slowly, the accuracy of the result may be rather low. For instance, for
        ``zipf(4)``, accuracy for mean, variance in example is only 1e-5.
        increasing `maxcount` and/or `chunksize` may improve the result,
        but may also make zipf very slow.

        The function is not vectorized.

        """
        if func is None:

            def fun(x):
                return (x + loc) * self._pmf(x, *args)
        else:

            def fun(x):
                return func(x + loc) * self._pmf(x, *args)
        _a, _b = self._get_support(*args)
        if lb is None:
            lb = _a
        else:
            lb = lb - loc
        if ub is None:
            ub = _b
        else:
            ub = ub - loc
        if conditional:
            invfac = self.sf(lb - 1, *args) - self.sf(ub, *args)
        else:
            invfac = 1.0
        if isinstance(self, rv_sample):
            res = self._expect(fun, lb, ub)
            return res / invfac
        x0 = self.ppf(0.5, *args)
        res = _expect(fun, lb, ub, x0, self.inc, maxcount, tolerance, chunksize)
        return res / invfac

    def _param_info(self):
        shape_info = self._shape_info()
        loc_info = _ShapeInfo('loc', True, (-np.inf, np.inf), (False, False))
        param_info = shape_info + [loc_info]
        return param_info