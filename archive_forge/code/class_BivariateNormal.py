import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
class BivariateNormal:

    def __init__(self, mean, cov):
        self.mean = mu
        self.cov = cov
        self.sigmax, self.sigmaxy, tmp, self.sigmay = np.ravel(cov)
        self.nvars = 2

    def rvs(self, size=1):
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def pdf(self, x):
        return bivariate_normal(x, self.mean, self.cov)

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        return self.expect(upper=x)

    def expect(self, func=lambda x: 1, lower=(-10, -10), upper=(10, 10)):

        def fun(x, y):
            x = np.column_stack((x, y))
            return func(x) * self.pdf(x)
        from scipy.integrate import dblquad
        return dblquad(fun, lower[0], upper[0], lambda y: lower[1], lambda y: upper[1])

    def kl(self, other):
        """Kullback-Leibler divergence between this and another distribution

        int f(x) (log f(x) - log g(x)) dx

        where f is the pdf of self, and g is the pdf of other

        uses double integration with scipy.integrate.dblquad

        limits currently hardcoded

        """
        fun = lambda x: self.logpdf(x) - other.logpdf(x)
        return self.expect(fun)

    def kl_mc(self, other, size=500000):
        fun = lambda x: self.logpdf(x) - other.logpdf(x)
        rvs = self.rvs(size=size)
        return fun(rvs).mean()