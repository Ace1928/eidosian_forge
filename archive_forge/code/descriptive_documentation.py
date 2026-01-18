import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils

        Returns the confidence intervals for the correlation coefficient

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value the upper confidence limit can be.
            Default is  99% confidence limit assuming normality.

        lower_bound : float
            Minimum value the lower confidence limit can be.
            Default is 99% confidence limit assuming normality.

        Returns
        -------
        interval : tuple
            Confidence interval for the correlation
        