import numpy as np
from scipy import stats
class ParametricMixtureD:
    """mixtures with a discrete distribution

    The mixing distribution is a discrete distribution like scipy.stats.poisson.
    All distribution in the mixture of the same type and parametrized
    by the outcome of the mixing distribution and have to be a continuous
    distribution (or have a pdf method).
    As an example, a mixture of normal distributed random variables with
    Poisson as the mixing distribution.


    assumes vectorized shape, loc and scale as in scipy.stats.distributions

    assume mixing_dist is frozen

    initialization looks fragile for all possible cases of lower and upper
    bounds of the distributions.

    """

    def __init__(self, mixing_dist, base_dist, bd_args_func, bd_kwds_func, cutoff=0.001):
        """create a mixture distribution

        Parameters
        ----------
        mixing_dist : discrete frozen distribution
            mixing distribution
        base_dist : continuous distribution
            parametrized distributions in the mixture
        bd_args_func : callable
            function that builds the tuple of args for the base_dist.
            The function obtains as argument the values in the support of
            the mixing distribution and should return an empty tuple or
            a tuple of arrays.
        bd_kwds_func : callable
            function that builds the dictionary of kwds for the base_dist.
            The function obtains as argument the values in the support of
            the mixing distribution and should return an empty dictionary or
            a dictionary with arrays as values.
        cutoff : float
            If the mixing distribution has infinite support, then the
            distribution is truncated with approximately (subject to integer
            conversion) the cutoff probability in the missing tail. Random
            draws that are outside the truncated range are clipped, that is
            assigned to the highest or lowest value in the truncated support.

        """
        self.mixing_dist = mixing_dist
        self.base_dist = base_dist
        if not np.isneginf(mixing_dist.dist.a):
            lower = mixing_dist.dist.a
        else:
            lower = mixing_dist.ppf(0.0001)
        if not np.isposinf(mixing_dist.dist.b):
            upper = mixing_dist.dist.b
        else:
            upper = mixing_dist.isf(0.0001)
        self.ma = lower
        self.mb = upper
        mixing_support = np.arange(lower, upper + 1)
        self.mixing_probs = mixing_dist.pmf(mixing_support)
        self.bd_args = bd_args_func(mixing_support)
        self.bd_kwds = bd_kwds_func(mixing_support)

    def rvs(self, size=1):
        mrvs = self.mixing_dist.rvs(size)
        mrvs_idx = (np.clip(mrvs, self.ma, self.mb) - self.ma).astype(int)
        bd_args = tuple((md[mrvs_idx] for md in self.bd_args))
        bd_kwds = {k: self.bd_kwds[k][mrvs_idx] for k in self.bd_kwds}
        kwds = {'size': size}
        kwds.update(bd_kwds)
        rvs = self.base_dist.rvs(*self.bd_args, **kwds)
        return (rvs, mrvs_idx)

    def pdf(self, x):
        x = np.asarray(x)
        if np.size(x) > 1:
            x = x[..., None]
        bd_probs = self.base_dist.pdf(x, *self.bd_args, **self.bd_kwds)
        prob = (bd_probs * self.mixing_probs).sum(-1)
        return (prob, bd_probs)

    def cdf(self, x):
        x = np.asarray(x)
        if np.size(x) > 1:
            x = x[..., None]
        bd_probs = self.base_dist.cdf(x, *self.bd_args, **self.bd_kwds)
        prob = (bd_probs * self.mixing_probs).sum(-1)
        return (prob, bd_probs)