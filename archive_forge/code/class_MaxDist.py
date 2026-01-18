from scipy import stats
class MaxDist(stats.rv_continuous):
    """ max of n of scipy.stats normal expon ...
        Example:
            maxnormal10 = RVmax( scipy.stats.norm, 10 )
            sample = maxnormal10( size=1000 )
            sample.cdf = cdf ^ n,  ppf ^ (1/n)
    """

    def __init__(self, dist, n):
        self.dist = dist
        self.n = n
        extradoc = 'maximumdistribution is the distribution of the ' + 'maximum of n i.i.d. random variable'
        super().__init__(name='maxdist', a=dist.a, b=dist.b, longname='A maximumdistribution')

    def _pdf(self, x, *args, **kw):
        return self.n * self.dist.pdf(x, *args, **kw) * self.dist.cdf(x, *args, **kw) ** (self.n - 1)

    def _cdf(self, x, *args, **kw):
        return self.dist.cdf(x, *args, **kw) ** self.n

    def _ppf(self, q, *args, **kw):
        return self.dist.ppf(q ** (1.0 / self.n), *args, **kw)