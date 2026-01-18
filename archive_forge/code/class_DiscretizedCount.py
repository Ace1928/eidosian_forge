import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel
class DiscretizedCount(rv_discrete):
    """Count distribution based on discretized distribution

    Parameters
    ----------
    distr : distribution instance
    d_offset : float
        Offset for integer interval, default is zero.
        The discrete random variable is ``y = floor(x + offset)`` where x is
        the continuous random variable.
        Warning: not verified for all methods.
    add_scale : bool
        If True (default), then the scale of the base distribution is added
        as parameter for the discrete distribution. The scale parameter is in
        the last position.
    kwds : keyword arguments
        The extra keyword arguments are used delegated to the ``__init__`` of
        the super class.
        Their usage has not been checked, e.g. currently the support of the
        distribution is assumed to be all non-negative integers.

    Notes
    -----
    `loc` argument is currently not supported, scale is not available for
    discrete distributions in scipy. The scale parameter of the underlying
    continuous distribution is the last shape parameter in this
    DiscretizedCount distribution if ``add_scale`` is True.

    The implementation was based mainly on [1]_ and [2]_. However, many new
    discrete distributions have been developed based on the approach that we
    use here. Note, that in many cases authors reparameterize the distribution,
    while this class inherits the parameterization from the underlying
    continuous distribution.

    References
    ----------
    .. [1] Chakraborty, Subrata, and Dhrubajyoti Chakravarty. "Discrete gamma
       distributions: Properties and parameter estimations." Communications in
       Statistics-Theory and Methods 41, no. 18 (2012): 3301-3324.

    .. [2] Alzaatreh, Ayman, Carl Lee, and Felix Famoye. 2012. “On the Discrete
       Analogues of Continuous Distributions.” Statistical Methodology 9 (6):
       589–603.


    """

    def __new__(cls, *args, **kwds):
        return super(rv_discrete, cls).__new__(cls)

    def __init__(self, distr, d_offset=0, add_scale=True, **kwds):
        self.distr = distr
        self.d_offset = d_offset
        self._ctor_param = distr._ctor_param
        self.add_scale = add_scale
        if distr.shapes is not None:
            self.k_shapes = len(distr.shapes.split(','))
            if add_scale:
                kwds.update({'shapes': distr.shapes + ', s'})
                self.k_shapes += 1
        elif add_scale:
            kwds.update({'shapes': 's'})
            self.k_shapes = 1
        else:
            self.k_shapes = 0
        super().__init__(**kwds)

    def _updated_ctor_param(self):
        dic = super()._updated_ctor_param()
        dic['distr'] = self.distr
        return dic

    def _unpack_args(self, args):
        if self.add_scale:
            scale = args[-1]
            args = args[:-1]
        else:
            scale = 1
        return (args, scale)

    def _rvs(self, *args, size=None, random_state=None):
        args, scale = self._unpack_args(args)
        if size is None:
            size = getattr(self, '_size', 1)
        rv = np.trunc(self.distr.rvs(*args, scale=scale, size=size, random_state=random_state) + self.d_offset)
        return rv

    def _pmf(self, x, *args):
        distr = self.distr
        if self.d_offset != 0:
            x = x + self.d_offset
        args, scale = self._unpack_args(args)
        p = distr.sf(x, *args, scale=scale) - distr.sf(x + 1, *args, scale=scale)
        return p

    def _cdf(self, x, *args):
        distr = self.distr
        args, scale = self._unpack_args(args)
        if self.d_offset != 0:
            x = x + self.d_offset
        p = distr.cdf(x + 1, *args, scale=scale)
        return p

    def _sf(self, x, *args):
        distr = self.distr
        args, scale = self._unpack_args(args)
        if self.d_offset != 0:
            x = x + self.d_offset
        p = distr.sf(x + 1, *args, scale=scale)
        return p

    def _ppf(self, p, *args):
        distr = self.distr
        args, scale = self._unpack_args(args)
        qc = distr.ppf(p, *args, scale=scale)
        if self.d_offset != 0:
            qc = qc + self.d_offset
        q = np.floor(qc * (1 - 1e-15))
        return q

    def _isf(self, p, *args):
        distr = self.distr
        args, scale = self._unpack_args(args)
        qc = distr.isf(p, *args, scale=scale)
        if self.d_offset != 0:
            qc = qc + self.d_offset
        q = np.floor(qc * (1 - 1e-15))
        return q