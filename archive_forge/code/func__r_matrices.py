from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def _r_matrices(delta_y_1_T, y_lag1, delta_x):
    """Returns two ndarrays needed for parameter estimation as well as the
    calculation of standard errors.

    Parameters
    ----------
    delta_y_1_T : ndarray (neqs x nobs)
        The first differences of endog.
        `.. math:: (y_1, \\ldots, y_T) - (y_0, \\ldots, y_{T-1})
    y_lag1 : ndarray (neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        Endog of the previous period (lag 1).
        `.. math:: (y_0, \\ldots, y_{T-1})
    delta_x : ndarray (k_ar_diff*neqs x nobs)
        (dimensions assuming no deterministic terms are given)
        Lagged differenced endog, used as regressor for the short term
        equation.

    Returns
    -------
    result : tuple
        A tuple of two ndarrays. (See p. 292 in [1]_ for the definition of
        R_0 and R_1.)

    References
    ----------
    .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
    """
    nobs = y_lag1.shape[1]
    m = np.identity(nobs) - delta_x.T.dot(inv(delta_x.dot(delta_x.T))).dot(delta_x)
    r0 = delta_y_1_T.dot(m)
    r1 = y_lag1.dot(m)
    return (r0, r1)