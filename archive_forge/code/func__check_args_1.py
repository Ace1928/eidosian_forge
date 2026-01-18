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
def _check_args_1(endog, n_factor, corr, nobs):
    msg = 'Either endog or corr must be provided.'
    if endog is not None and corr is not None:
        raise ValueError(msg)
    if endog is None and corr is None:
        warnings.warn('Both endog and corr are provided, ' + 'corr will be used for factor analysis.')
    if n_factor <= 0:
        raise ValueError('n_factor must be larger than 0! %d < 0' % n_factor)
    if nobs is not None and endog is not None:
        warnings.warn('nobs is ignored when endog is provided')