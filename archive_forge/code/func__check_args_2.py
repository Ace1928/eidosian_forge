import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
def _check_args_2(endog, n_factor, corr, nobs, k_endog):
    if n_factor > k_endog:
        raise ValueError('n_factor cannot be greater than the number of variables! %d > %d' % (n_factor, k_endog))
    if np.max(np.abs(np.diag(corr) - 1)) > 1e-10:
        raise ValueError('corr must be a correlation matrix')
    if corr.shape[0] != corr.shape[1]:
        raise ValueError('Correlation matrix corr must be a square (rows %d != cols %d)' % corr.shape)