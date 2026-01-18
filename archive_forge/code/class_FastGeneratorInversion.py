import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
class FastGeneratorInversion:
    """
    Fast sampling by numerical inversion of the CDF for a large class of
    continuous distributions in `scipy.stats`.

    Parameters
    ----------
    dist : rv_frozen object
        Frozen distribution object from `scipy.stats`. The list of supported
        distributions can be found in the Notes section. The shape parameters,
        `loc` and `scale` used to create the distributions must be scalars.
        For example, for the Gamma distribution with shape parameter `p`,
        `p` has to be a float, and for the beta distribution with shape
        parameters (a, b), both a and b have to be floats.
    domain : tuple of floats, optional
        If one wishes to sample from a truncated/conditional distribution,
        the domain has to be specified.
        The default is None. In that case, the random variates are not
        truncated, and the domain is inferred from the support of the
        distribution.
    ignore_shape_range : boolean, optional.
        If False, shape parameters that are outside of the valid range
        of values to ensure that the numerical accuracy (see Notes) is
        high, raise a ValueError. If True, any shape parameters that are valid
        for the distribution are accepted. This can be useful for testing.
        The default is False.
    random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            A NumPy random number generator or seed for the underlying NumPy
            random number generator used to generate the stream of uniform
            random numbers.
            If `random_state` is None, it uses ``self.random_state``.
            If `random_state` is an int,
            ``np.random.default_rng(random_state)`` is used.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.

    Attributes
    ----------
    loc : float
        The location parameter.
    random_state : {`numpy.random.Generator`, `numpy.random.RandomState`}
        The random state used in relevant methods like `rvs` (unless
        another `random_state` is passed as an argument to these methods).
    scale : float
        The scale parameter.

    Methods
    -------
    cdf
    evaluate_error
    ppf
    qrvs
    rvs
    support

    Notes
    -----
    The class creates an object for continuous distributions specified
    by `dist`. The method `rvs` uses a generator from
    `scipy.stats.sampling` that is created when the object is instantiated.
    In addition, the methods `qrvs` and `ppf` are added.
    `qrvs` generate samples based on quasi-random numbers from
    `scipy.stats.qmc`. `ppf` is the PPF based on the
    numerical inversion method in [1]_ (`NumericalInversePolynomial`) that is
    used to generate random variates.

    Supported distributions (`distname`) are:
    ``alpha``, ``anglit``, ``argus``, ``beta``, ``betaprime``, ``bradford``,
    ``burr``, ``burr12``, ``cauchy``, ``chi``, ``chi2``, ``cosine``,
    ``crystalball``, ``expon``, ``gamma``, ``gennorm``, ``geninvgauss``,
    ``gumbel_l``, ``gumbel_r``, ``hypsecant``, ``invgamma``, ``invgauss``,
    ``invweibull``, ``laplace``, ``logistic``, ``maxwell``, ``moyal``,
    ``norm``, ``pareto``, ``powerlaw``, ``t``, ``rayleigh``, ``semicircular``,
    ``wald``, ``weibull_max``, ``weibull_min``.

    `rvs` relies on the accuracy of the numerical inversion. If very extreme
    shape parameters are used, the numerical inversion might not work. However,
    for all implemented distributions, the admissible shape parameters have
    been tested, and an error will be raised if the user supplies values
    outside of the allowed range. The u-error should not exceed 1e-10 for all
    valid parameters. Note that warnings might be raised even if parameters
    are within the valid range when the object is instantiated.
    To check numerical accuracy, the method `evaluate_error` can be used.

    Note that all implemented distributions are also part of `scipy.stats`, and
    the object created by `FastGeneratorInversion` relies on methods like
    `ppf`, `cdf` and `pdf` from `rv_frozen`. The main benefit of using this
    class can be summarized as follows: Once the generator to sample random
    variates is created in the setup step, sampling and evaluation of
    the PPF using `ppf` are very fast,
    and performance is essentially independent of the distribution. Therefore,
    a substantial speed-up can be achieved for many distributions if large
    numbers of random variates are required. It is important to know that this
    fast sampling is achieved by inversion of the CDF. Thus, one uniform
    random variate is transformed into a non-uniform variate, which is an
    advantage for several simulation methods, e.g., when
    the variance reduction methods of common random variates or
    antithetic variates are be used ([2]_).

    In addition, inversion makes it possible to
    - to use a QMC generator from `scipy.stats.qmc` (method `qrvs`),
    - to generate random variates truncated to an interval. For example, if
    one aims to sample standard normal random variates from
    the interval (2, 4), this can be easily achieved by using the parameter
    `domain`.

    The location and scale that are initially defined by `dist`
    can be reset without having to rerun the setup
    step to create the generator that is used for sampling. The relation
    of the distribution `Y` with `loc` and `scale` to the standard
    distribution `X` (i.e., ``loc=0`` and ``scale=1``) is given by
    ``Y = loc + scale * X``.

    References
    ----------
    .. [1] Derflinger, Gerhard, Wolfgang Hörmann, and Josef Leydold.
           "Random variate  generation by numerical inversion when only the
           density is known." ACM Transactions on Modeling and Computer
           Simulation (TOMACS) 20.4 (2010): 1-25.
    .. [2] Hörmann, Wolfgang, Josef Leydold and Gerhard Derflinger.
           "Automatic nonuniform random number generation."
           Springer, 2004.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from scipy.stats.sampling import FastGeneratorInversion

    Let's start with a simple example to illustrate the main features:

    >>> gamma_frozen = stats.gamma(1.5)
    >>> gamma_dist = FastGeneratorInversion(gamma_frozen)
    >>> r = gamma_dist.rvs(size=1000)

    The mean should be approximately equal to the shape parameter 1.5:

    >>> r.mean()
    1.52423591130436  # may vary

    Similarly, we can draw a sample based on quasi-random numbers:

    >>> r = gamma_dist.qrvs(size=1000)
    >>> r.mean()
    1.4996639255942914  # may vary

    Compare the PPF against approximation `ppf`.

    >>> q = [0.001, 0.2, 0.5, 0.8, 0.999]
    >>> np.max(np.abs(gamma_frozen.ppf(q) - gamma_dist.ppf(q)))
    4.313394796895409e-08

    To confirm that the numerical inversion is accurate, we evaluate the
    approximation error (u-error), which should be below 1e-10 (for more
    details, refer to the documentation of `evaluate_error`):

    >>> gamma_dist.evaluate_error()
    (7.446320551265581e-11, nan)  # may vary

    Note that the location and scale can be changed without instantiating a
    new generator:

    >>> gamma_dist.loc = 2
    >>> gamma_dist.scale = 3
    >>> r = gamma_dist.rvs(size=1000)

    The mean should be approximately 2 + 3*1.5 = 6.5.

    >>> r.mean()
    6.399549295242894  # may vary

    Let us also illustrate how truncation can be applied:

    >>> trunc_norm = FastGeneratorInversion(stats.norm(), domain=(3, 4))
    >>> r = trunc_norm.rvs(size=1000)
    >>> 3 < r.min() < r.max() < 4
    True

    Check the mean:

    >>> r.mean()
    3.250433367078603  # may vary

    >>> stats.norm.expect(lb=3, ub=4, conditional=True)
    3.260454285589997

    In this particular, case, `scipy.stats.truncnorm` could also be used to
    generate truncated normal random variates.

    """

    def __init__(self, dist, *, domain=None, ignore_shape_range=False, random_state=None):
        if isinstance(dist, stats.distributions.rv_frozen):
            distname = dist.dist.name
            if distname not in PINV_CONFIG.keys():
                raise ValueError(f"Distribution '{distname}' is not supported.It must be one of {list(PINV_CONFIG.keys())}")
        else:
            raise ValueError('`dist` must be a frozen distribution object')
        loc = dist.kwds.get('loc', 0)
        scale = dist.kwds.get('scale', 1)
        args = dist.args
        if not np.isscalar(loc):
            raise ValueError('loc must be scalar.')
        if not np.isscalar(scale):
            raise ValueError('scale must be scalar.')
        self._frozendist = getattr(stats, distname)(*args, loc=loc, scale=scale)
        self._distname = distname
        nargs = np.broadcast_arrays(args)[0].size
        nargs_expected = self._frozendist.dist.numargs
        if nargs != nargs_expected:
            raise ValueError(f'Each of the {nargs_expected} shape parameters must be a scalar, but {nargs} values are provided.')
        self.random_state = random_state
        if domain is None:
            self._domain = self._frozendist.support()
            self._p_lower = 0.0
            self._p_domain = 1.0
        else:
            self._domain = domain
            self._p_lower = self._frozendist.cdf(self._domain[0])
            _p_domain = self._frozendist.cdf(self._domain[1]) - self._p_lower
            self._p_domain = _p_domain
        self._set_domain_adj()
        self._ignore_shape_range = ignore_shape_range
        self._domain_pinv = self._domain
        dist = self._process_config(distname, args)
        if self._rvs_transform_inv is not None:
            d0 = self._rvs_transform_inv(self._domain[0], *args)
            d1 = self._rvs_transform_inv(self._domain[1], *args)
            if d0 > d1:
                d0, d1 = (d1, d0)
            self._domain_pinv = (d0, d1)
        if self._center is not None:
            if self._center < self._domain_pinv[0]:
                self._center = self._domain_pinv[0]
            elif self._center > self._domain_pinv[1]:
                self._center = self._domain_pinv[1]
        self._rng = NumericalInversePolynomial(dist, random_state=self.random_state, domain=self._domain_pinv, center=self._center)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = check_random_state_qmc(random_state)

    @property
    def loc(self):
        return self._frozendist.kwds.get('loc', 0)

    @loc.setter
    def loc(self, loc):
        if not np.isscalar(loc):
            raise ValueError('loc must be scalar.')
        self._frozendist.kwds['loc'] = loc
        self._set_domain_adj()

    @property
    def scale(self):
        return self._frozendist.kwds.get('scale', 0)

    @scale.setter
    def scale(self, scale):
        if not np.isscalar(scale):
            raise ValueError('scale must be scalar.')
        self._frozendist.kwds['scale'] = scale
        self._set_domain_adj()

    def _set_domain_adj(self):
        """ Adjust the domain based on loc and scale. """
        loc = self.loc
        scale = self.scale
        lb = self._domain[0] * scale + loc
        ub = self._domain[1] * scale + loc
        self._domain_adj = (lb, ub)

    def _process_config(self, distname, args):
        cfg = PINV_CONFIG[distname]
        if 'check_pinv_params' in cfg:
            if not self._ignore_shape_range:
                if not cfg['check_pinv_params'](*args):
                    msg = f'No generator is defined for the shape parameters {args}. Use ignore_shape_range to proceed with the selected values.'
                    raise ValueError(msg)
        if 'center' in cfg.keys():
            if not np.isscalar(cfg['center']):
                self._center = cfg['center'](*args)
            else:
                self._center = cfg['center']
        else:
            self._center = None
        self._rvs_transform = cfg.get('rvs_transform', None)
        self._rvs_transform_inv = cfg.get('rvs_transform_inv', None)
        _mirror_uniform = cfg.get('mirror_uniform', None)
        if _mirror_uniform is None:
            self._mirror_uniform = False
        else:
            self._mirror_uniform = _mirror_uniform(*args)
        return CustomDistPINV(cfg['pdf'], args)

    def rvs(self, size=None):
        """
        Sample from the distribution by inversion.

        Parameters
        ----------
        size : int or tuple, optional
            The shape of samples. Default is ``None`` in which case a scalar
            sample is returned.

        Returns
        -------
        rvs : array_like
            A NumPy array of random variates.

        Notes
        -----
        Random variates are generated by numerical inversion of the CDF, i.e.,
        `ppf` computed by `NumericalInversePolynomial` when the class
        is instantiated. Note that the
        default ``rvs`` method of the rv_continuous class is
        overwritten. Hence, a different stream of random numbers is generated
        even if the same seed is used.
        """
        u = self.random_state.uniform(size=size)
        if self._mirror_uniform:
            u = 1 - u
        r = self._rng.ppf(u)
        if self._rvs_transform is not None:
            r = self._rvs_transform(r, *self._frozendist.args)
        return self.loc + self.scale * r

    def ppf(self, q):
        """
        Very fast PPF (inverse CDF) of the distribution which
        is a very close approximation of the exact PPF values.

        Parameters
        ----------
        u : array_like
            Array with probabilities.

        Returns
        -------
        ppf : array_like
            Quantiles corresponding to the values in `u`.

        Notes
        -----
        The evaluation of the PPF is very fast but it may have a large
        relative error in the far tails. The numerical precision of the PPF
        is controlled by the u-error, that is,
        ``max |u - CDF(PPF(u))|`` where the max is taken over points in
        the interval [0,1], see `evaluate_error`.

        Note that this PPF is designed to generate random samples.
        """
        q = np.asarray(q)
        if self._mirror_uniform:
            x = self._rng.ppf(1 - q)
        else:
            x = self._rng.ppf(q)
        if self._rvs_transform is not None:
            x = self._rvs_transform(x, *self._frozendist.args)
        return self.scale * x + self.loc

    def qrvs(self, size=None, d=None, qmc_engine=None):
        """
        Quasi-random variates of the given distribution.

        The `qmc_engine` is used to draw uniform quasi-random variates, and
        these are converted to quasi-random variates of the given distribution
        using inverse transform sampling.

        Parameters
        ----------
        size : int, tuple of ints, or None; optional
            Defines shape of random variates array. Default is ``None``.
        d : int or None, optional
            Defines dimension of uniform quasi-random variates to be
            transformed. Default is ``None``.
        qmc_engine : scipy.stats.qmc.QMCEngine(d=1), optional
            Defines the object to use for drawing
            quasi-random variates. Default is ``None``, which uses
            `scipy.stats.qmc.Halton(1)`.

        Returns
        -------
        rvs : ndarray or scalar
            Quasi-random variates. See Notes for shape information.

        Notes
        -----
        The shape of the output array depends on `size`, `d`, and `qmc_engine`.
        The intent is for the interface to be natural, but the detailed rules
        to achieve this are complicated.

        - If `qmc_engine` is ``None``, a `scipy.stats.qmc.Halton` instance is
          created with dimension `d`. If `d` is not provided, ``d=1``.
        - If `qmc_engine` is not ``None`` and `d` is ``None``, `d` is
          determined from the dimension of the `qmc_engine`.
        - If `qmc_engine` is not ``None`` and `d` is not ``None`` but the
          dimensions are inconsistent, a ``ValueError`` is raised.
        - After `d` is determined according to the rules above, the output
          shape is ``tuple_shape + d_shape``, where:

              - ``tuple_shape = tuple()`` if `size` is ``None``,
              - ``tuple_shape = (size,)`` if `size` is an ``int``,
              - ``tuple_shape = size`` if `size` is a sequence,
              - ``d_shape = tuple()`` if `d` is ``None`` or `d` is 1, and
              - ``d_shape = (d,)`` if `d` is greater than 1.

        The elements of the returned array are part of a low-discrepancy
        sequence. If `d` is 1, this means that none of the samples are truly
        independent. If `d` > 1, each slice ``rvs[..., i]`` will be of a
        quasi-independent sequence; see `scipy.stats.qmc.QMCEngine` for
        details. Note that when `d` > 1, the samples returned are still those
        of the provided univariate distribution, not a multivariate
        generalization of that distribution.

        """
        qmc_engine, d = _validate_qmc_input(qmc_engine, d, self.random_state)
        try:
            if size is None:
                tuple_size = (1,)
            else:
                tuple_size = tuple(size)
        except TypeError:
            tuple_size = (size,)
        N = 1 if size is None else np.prod(size)
        u = qmc_engine.random(N)
        if self._mirror_uniform:
            u = 1 - u
        qrvs = self._ppf(u)
        if self._rvs_transform is not None:
            qrvs = self._rvs_transform(qrvs, *self._frozendist.args)
        if size is None:
            qrvs = qrvs.squeeze()[()]
        elif d == 1:
            qrvs = qrvs.reshape(tuple_size)
        else:
            qrvs = qrvs.reshape(tuple_size + (d,))
        return self.loc + self.scale * qrvs

    def evaluate_error(self, size=100000, random_state=None, x_error=False):
        """
        Evaluate the numerical accuracy of the inversion (u- and x-error).

        Parameters
        ----------
        size : int, optional
            The number of random points over which the error is estimated.
            Default is ``100000``.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            A NumPy random number generator or seed for the underlying NumPy
            random number generator used to generate the stream of uniform
            random numbers.
            If `random_state` is None, use ``self.random_state``.
            If `random_state` is an int,
            ``np.random.default_rng(random_state)`` is used.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.

        Returns
        -------
        u_error, x_error : tuple of floats
            A NumPy array of random variates.

        Notes
        -----
        The numerical precision of the inverse CDF `ppf` is controlled by
        the u-error. It is computed as follows:
        ``max |u - CDF(PPF(u))|`` where the max is taken `size` random
        points in the interval [0,1]. `random_state` determines the random
        sample. Note that if `ppf` was exact, the u-error would be zero.

        The x-error measures the direct distance between the exact PPF
        and `ppf`. If ``x_error`` is set to ``True`, it is
        computed as the maximum of the minimum of the relative and absolute
        x-error:
        ``max(min(x_error_abs[i], x_error_rel[i]))`` where
        ``x_error_abs[i] = |PPF(u[i]) - PPF_fast(u[i])|``,
        ``x_error_rel[i] = max |(PPF(u[i]) - PPF_fast(u[i])) / PPF(u[i])|``.
        Note that it is important to consider the relative x-error in the case
        that ``PPF(u)`` is close to zero or very large.

        By default, only the u-error is evaluated and the x-error is set to
        ``np.nan``. Note that the evaluation of the x-error will be very slow
        if the implementation of the PPF is slow.

        Further information about these error measures can be found in [1]_.

        References
        ----------
        .. [1] Derflinger, Gerhard, Wolfgang Hörmann, and Josef Leydold.
               "Random variate  generation by numerical inversion when only the
               density is known." ACM Transactions on Modeling and Computer
               Simulation (TOMACS) 20.4 (2010): 1-25.

        Examples
        --------

        >>> import numpy as np
        >>> from scipy import stats
        >>> from scipy.stats.sampling import FastGeneratorInversion

        Create an object for the normal distribution:

        >>> d_norm_frozen = stats.norm()
        >>> d_norm = FastGeneratorInversion(d_norm_frozen)

        To confirm that the numerical inversion is accurate, we evaluate the
        approximation error (u-error and x-error).

        >>> u_error, x_error = d_norm.evaluate_error(x_error=True)

        The u-error should be below 1e-10:

        >>> u_error
        8.785783212061915e-11  # may vary

        Compare the PPF against approximation `ppf`:

        >>> q = [0.001, 0.2, 0.4, 0.6, 0.8, 0.999]
        >>> diff = np.abs(d_norm_frozen.ppf(q) - d_norm.ppf(q))
        >>> x_error_abs = np.max(diff)
        >>> x_error_abs
        1.2937954707581412e-08

        This is the absolute x-error evaluated at the points q. The relative
        error is given by

        >>> x_error_rel = np.max(diff / np.abs(d_norm_frozen.ppf(q)))
        >>> x_error_rel
        4.186725600453555e-09

        The x_error computed above is derived in a very similar way over a
        much larger set of random values q. At each value q[i], the minimum
        of the relative and absolute error is taken. The final value is then
        derived as the maximum of these values. In our example, we get the
        following value:

        >>> x_error
        4.507068014335139e-07  # may vary

        """
        if not isinstance(size, (numbers.Integral, np.integer)):
            raise ValueError('size must be an integer.')
        urng = check_random_state_qmc(random_state)
        u = urng.uniform(size=size)
        if self._mirror_uniform:
            u = 1 - u
        x = self.ppf(u)
        uerr = np.max(np.abs(self._cdf(x) - u))
        if not x_error:
            return (uerr, np.nan)
        ppf_u = self._ppf(u)
        x_error_abs = np.abs(self.ppf(u) - ppf_u)
        x_error_rel = x_error_abs / np.abs(ppf_u)
        x_error_combined = np.array([x_error_abs, x_error_rel]).min(axis=0)
        return (uerr, np.max(x_error_combined))

    def support(self):
        """Support of the distribution.

        Returns
        -------
        a, b : float
            end-points of the distribution's support.

        Notes
        -----

        Note that the support of the distribution depends on `loc`,
        `scale` and `domain`.

        Examples
        --------

        >>> from scipy import stats
        >>> from scipy.stats.sampling import FastGeneratorInversion

        Define a truncated normal distribution:

        >>> d_norm = FastGeneratorInversion(stats.norm(), domain=(0, 1))
        >>> d_norm.support()
        (0, 1)

        Shift the distribution:

        >>> d_norm.loc = 2.5
        >>> d_norm.support()
        (2.5, 3.5)

        """
        return self._domain_adj

    def _cdf(self, x):
        """Cumulative distribution function (CDF)

        Parameters
        ----------
        x : array_like
            The values where the CDF is evaluated

        Returns
        -------
        y : ndarray
            CDF evaluated at x

        """
        y = self._frozendist.cdf(x)
        if self._p_domain == 1.0:
            return y
        return np.clip((y - self._p_lower) / self._p_domain, 0, 1)

    def _ppf(self, q):
        """Percent point function (inverse of `cdf`)

        Parameters
        ----------
        q : array_like
            lower tail probability

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.

        """
        if self._p_domain == 1.0:
            return self._frozendist.ppf(q)
        x = self._frozendist.ppf(self._p_domain * np.array(q) + self._p_lower)
        return np.clip(x, self._domain_adj[0], self._domain_adj[1])