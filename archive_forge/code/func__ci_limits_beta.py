import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts
def _ci_limits_beta(self, b0, param_num=None):
    """
        Returns the difference between the log likelihood for a
        parameter and some critical value.

        Parameters
        ----------
        b0: float
            Value of a regression parameter
        param_num : int
            Parameter index of b0
        """
    return self.test_beta([b0], [param_num])[0] - self.r0