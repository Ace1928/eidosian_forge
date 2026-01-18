import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
class MultivariateNormalChol:
    """multivariate normal distribution with cholesky decomposition of sigma

    ignoring mean at the beginning, maybe

    needs testing for broadcasting to contemporaneously but not intertemporaly
    correlated random variable, which axis?,
    maybe swapaxis or rollaxis if x.ndim != mean.ndim == (sigma.ndim - 1)

    initially 1d is ok, 2d should work with iid in axis 0 and mvn in axis 1

    """

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        self.sigmainv = sigmainv
        self.cholsigma = linalg.cholesky(sigma)
        self.cholsigmainv = linalg.cholesky(sigmainv)[::-1, ::-1]

    def whiten(self, x):
        return np.dot(cholsigmainv, x)

    def logpdf_obs(self, x):
        x = x - self.mean
        x_whitened = self.whiten(x)
        logdetsigma = np.log(np.linalg.det(sigma))
        sigma2 = 1.0
        llike = 0.5 * (np.log(sigma2) - 2.0 * np.log(np.diagonal(self.cholsigmainv)) + x_whitened ** 2 / sigma2 + np.log(2 * np.pi))
        return llike

    def logpdf(self, x):
        return self.logpdf_obs(x).sum(-1)

    def pdf(self, x):
        return np.exp(self.logpdf(x))