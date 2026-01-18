from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def _est_regularized_naive(mod, pnum, partitions, fit_kwds=None):
    """estimates the regularized fitted parameters.

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    pnum : scalar
        Index of current partition
    partitions : scalar
        Total number of partitions
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit_regularized

    Returns
    -------
    An array of the parameters for the regularized fit
    """
    if fit_kwds is None:
        raise ValueError('_est_regularized_naive currently ' + 'requires that fit_kwds not be None.')
    return mod.fit_regularized(**fit_kwds).params