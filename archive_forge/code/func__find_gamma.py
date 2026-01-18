import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _find_gamma(self, gamma):
    """
        Finds gamma that satisfies
        sum(log(n * w(gamma))) - log(r0) = 0

        Used for confidence intervals for the mean

        Parameters
        ----------
        gamma : float
            Lagrange multiplier when computing confidence interval

        Returns
        -------
        diff : float
            The difference between the log-liklihood when the Lagrange
            multiplier is gamma and a pre-specified value
        """
    denom = np.sum((self.endog - gamma) ** (-1))
    new_weights = (self.endog - gamma) ** (-1) / denom
    return -2 * np.sum(np.log(self.nobs * new_weights)) - self.r0