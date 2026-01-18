import numpy as np
import warnings
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
def _parzen(u):
    z = np.where(np.abs(u) <= 0.5, 4.0 / 3 - 8.0 * u ** 2 + 8.0 * np.abs(u) ** 3, 8.0 * (1 - np.abs(u)) ** 3 / 3.0)
    z[np.abs(u) > 1] = 0
    return z