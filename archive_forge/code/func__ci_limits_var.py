import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def _ci_limits_var(self, var):
    """
        Used to determine the confidence intervals for the variance.
        It calls test_var and when called by an optimizer,
        finds the value of sig2_0 that is chi2.ppf(significance-level)

        Parameters
        ----------
        var_test : float
            Hypothesized value of the variance

        Returns
        -------
        diff : float
            The difference between the log likelihood ratio at var_test and a
            pre-specified value.
        """
    return self.test_var(var)[0] - self.r0