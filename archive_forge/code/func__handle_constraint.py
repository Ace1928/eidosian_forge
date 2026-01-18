from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def _handle_constraint(self, mean_params, bcov):
    """
        Expand the parameter estimate `mean_params` and covariance matrix
        `bcov` to the coordinate system of the unconstrained model.

        Parameters
        ----------
        mean_params : array_like
            A parameter vector estimate for the reduced model.
        bcov : array_like
            The covariance matrix of mean_params.

        Returns
        -------
        mean_params : array_like
            The input parameter vector mean_params, expanded to the
            coordinate system of the full model
        bcov : array_like
            The input covariance matrix bcov, expanded to the
            coordinate system of the full model
        """
    red_p = len(mean_params)
    full_p = self.constraint.lhs.shape[1]
    mean_params0 = np.r_[mean_params, np.zeros(full_p - red_p)]
    save_exog_li = self.exog_li
    self.exog_li = self.constraint.exog_fulltrans_li
    import copy
    save_cached_means = copy.deepcopy(self.cached_means)
    self.update_cached_means(mean_params0)
    _, score = self._update_mean_params()
    if score is None:
        warnings.warn('Singular matrix encountered in GEE score test', ConvergenceWarning)
        return (None, None)
    _, ncov1, cmat = self._covmat()
    scale = self.estimate_scale()
    cmat = cmat / scale ** 2
    score2 = score[red_p:] / scale
    amat = np.linalg.inv(ncov1)
    bmat_11 = cmat[0:red_p, 0:red_p]
    bmat_22 = cmat[red_p:, red_p:]
    bmat_12 = cmat[0:red_p, red_p:]
    amat_11 = amat[0:red_p, 0:red_p]
    amat_12 = amat[0:red_p, red_p:]
    score_cov = bmat_22 - np.dot(amat_12.T, np.linalg.solve(amat_11, bmat_12))
    score_cov -= np.dot(bmat_12.T, np.linalg.solve(amat_11, amat_12))
    score_cov += np.dot(amat_12.T, np.dot(np.linalg.solve(amat_11, bmat_11), np.linalg.solve(amat_11, amat_12)))
    from scipy.stats.distributions import chi2
    score_statistic = np.dot(score2, np.linalg.solve(score_cov, score2))
    score_df = len(score2)
    score_pvalue = 1 - chi2.cdf(score_statistic, score_df)
    self.score_test_results = {'statistic': score_statistic, 'df': score_df, 'p-value': score_pvalue}
    mean_params = self.constraint.unpack_param(mean_params)
    bcov = self.constraint.unpack_cov(bcov)
    self.exog_li = save_exog_li
    self.cached_means = save_cached_means
    self.exog = self.constraint.restore_exog()
    return (mean_params, bcov)