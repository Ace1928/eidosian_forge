import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def _rep_data_gen(endog, exog, partitions):
    """partitions data"""
    n_exog = exog.shape[0]
    n_part = np.ceil(n_exog / partitions)
    ii = 0
    while ii < n_exog:
        yield (endog, exog)
        ii += int(n_part)