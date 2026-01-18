import numpy as np
from numpy.testing import assert_almost_equal, assert_
import scipy
from scipy import stats
import matplotlib.pylab as plt
class gaussian_kde_covfact(stats.gaussian_kde):

    def __init__(self, dataset, covfact='scotts'):
        self.covfact = covfact
        scipy.stats.gaussian_kde.__init__(self, dataset)

    def _compute_covariance_(self):
        """not used"""
        self.inv_cov = np.linalg.inv(self.covariance)
        self._norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance)) * self.n

    def covariance_factor(self):
        if self.covfact in ['sc', 'scotts']:
            return self.scotts_factor()
        if self.covfact in ['si', 'silverman']:
            return self.silverman_factor()
        elif self.covfact:
            return float(self.covfact)
        else:
            raise ValueError('covariance factor has to be scotts, silverman or a number')

    def reset_covfact(self, covfact):
        self.covfact = covfact
        self.covariance_factor()
        self._compute_covariance()