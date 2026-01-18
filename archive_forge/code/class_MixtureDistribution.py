import numpy as np
class MixtureDistribution:
    """univariate mixture distribution

    for simple case for now (unbound support)
    does not yet inherit from scipy.stats.distributions

    adding pdf to mixture_rvs, some restrictions on broadcasting
    Currently it does not hold any state, all arguments included in each method.
    """

    def rvs(self, prob, size, dist, kwargs=None):
        return mixture_rvs(prob, size, dist, kwargs=kwargs)

    def pdf(self, x, prob, dist, kwargs=None):
        """
        pdf a mixture of distributions.

        Parameters
        ----------
        x : array_like
            Array containing locations where the PDF should be evaluated
        prob : array_like
            Probability of sampling from each distribution in dist
        dist : array_like
            An iterable of distributions objects from scipy.stats.
        kwargs : tuple of dicts, optional
            A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and
            args to be passed to the respective distribution in dist.  If not
            provided, the distribution defaults are used.

        Examples
        --------
        Say we want 5000 random variables from mixture of normals with two
        distributions norm(-1,.5) and norm(1,.5) and we want to sample from the
        first with probability .75 and the second with probability .25.

        >>> import numpy as np
        >>> from scipy import stats
        >>> from statsmodels.distributions.mixture_rvs import MixtureDistribution
        >>> x = np.arange(-4.0, 4.0, 0.01)
        >>> prob = [.75,.25]
        >>> mixture = MixtureDistribution()
        >>> Y = mixture.pdf(x, prob, dist=[stats.norm, stats.norm],
        ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
        """
        if len(prob) != len(dist):
            raise ValueError('You must provide as many probabilities as distributions')
        if not np.allclose(np.sum(prob), 1):
            raise ValueError('prob does not sum to 1')
        if kwargs is None:
            kwargs = ({},) * len(prob)
        for i in range(len(prob)):
            loc = kwargs[i].get('loc', 0)
            scale = kwargs[i].get('scale', 1)
            args = kwargs[i].get('args', ())
            if i == 0:
                pdf_ = prob[i] * dist[i].pdf(x, *args, loc=loc, scale=scale)
            else:
                pdf_ += prob[i] * dist[i].pdf(x, *args, loc=loc, scale=scale)
        return pdf_

    def cdf(self, x, prob, dist, kwargs=None):
        """
        cdf of a mixture of distributions.

        Parameters
        ----------
        x : array_like
            Array containing locations where the CDF should be evaluated
        prob : array_like
            Probability of sampling from each distribution in dist
        size : int
            The length of the returned sample.
        dist : array_like
            An iterable of distributions objects from scipy.stats.
        kwargs : tuple of dicts, optional
            A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and
            args to be passed to the respective distribution in dist.  If not
            provided, the distribution defaults are used.

        Examples
        --------
        Say we want 5000 random variables from mixture of normals with two
        distributions norm(-1,.5) and norm(1,.5) and we want to sample from the
        first with probability .75 and the second with probability .25.

        >>> import numpy as np
        >>> from scipy import stats
        >>> from statsmodels.distributions.mixture_rvs import MixtureDistribution
        >>> x = np.arange(-4.0, 4.0, 0.01)
        >>> prob = [.75,.25]
        >>> mixture = MixtureDistribution()
        >>> Y = mixture.pdf(x, prob, dist=[stats.norm, stats.norm],
        ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
        """
        if len(prob) != len(dist):
            raise ValueError('You must provide as many probabilities as distributions')
        if not np.allclose(np.sum(prob), 1):
            raise ValueError('prob does not sum to 1')
        if kwargs is None:
            kwargs = ({},) * len(prob)
        for i in range(len(prob)):
            loc = kwargs[i].get('loc', 0)
            scale = kwargs[i].get('scale', 1)
            args = kwargs[i].get('args', ())
            if i == 0:
                cdf_ = prob[i] * dist[i].cdf(x, *args, loc=loc, scale=scale)
            else:
                cdf_ += prob[i] * dist[i].cdf(x, *args, loc=loc, scale=scale)
        return cdf_