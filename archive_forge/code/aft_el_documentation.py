import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts

        Returns the confidence interval for a regression
        parameter in the AFT model.

        Parameters
        ----------
        param_num : int
            Parameter number of interest
        beta_high : float
            Upper bound for the confidence interval
        beta_low : float
            Lower bound for the confidence interval
        sig : float, optional
            Significance level.  Default is .05

        Notes
        -----
        If the function returns f(a) and f(b) must have different signs,
        consider widening the search area by adjusting beta_low and
        beta_high.

        Also note that this process is computational intensive.  There
        are 4 levels of optimization/solving.  From outer to inner:

        1) Solving so that llr-critical value = 0
        2) maximizing over nuisance parameters
        3) Using  EM at each value of nuisamce parameters
        4) Using the _modified_Newton optimizer at each iteration
           of the EM algorithm.

        Also, for very unlikely nuisance parameters, it is possible for
        the EM algorithm to not converge.  This is not an indicator
        that the solver did not find the correct solution.  It just means
        for a specific iteration of the nuisance parameters, the optimizer
        was unable to converge.

        If the user desires to verify the success of the optimization,
        it is recommended to test the limits using test_beta.
        