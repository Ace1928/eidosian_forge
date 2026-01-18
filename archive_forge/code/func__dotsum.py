import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _dotsum(x, y):
    """
    Returns sum(x * y), where '*' is the pointwise product, computed
    efficiently for dense and sparse matrices.
    """
    if sparse.issparse(x):
        return x.multiply(y).sum()
    else:
        return np.dot(x.ravel(), y.ravel())