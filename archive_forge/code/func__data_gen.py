import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def _data_gen(endog, exog, partitions):
    """partitions data"""
    n_exog = exog.shape[0]
    n_part = np.ceil(n_exog / partitions)
    n_part = np.floor(n_exog / partitions)
    rem = n_exog - n_part * partitions
    stp = 0
    while stp < partitions - 1:
        ii = int(n_part * stp)
        jj = int(n_part * (stp + 1))
        yield (endog[ii:jj], exog[ii:jj, :])
        stp += 1
    ii = int(n_part * stp)
    jj = int(n_part * (stp + 1) + rem)
    yield (endog[ii:jj], exog[ii:jj, :])