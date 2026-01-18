import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _ci_limits_skew(self, skew):
    """
        Parameters
        ----------
        skew0 : float
            Hypothesized value of skewness

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at skew and a
            pre-specified value.
        """
    return self.test_skew(skew)[0] - self.r0