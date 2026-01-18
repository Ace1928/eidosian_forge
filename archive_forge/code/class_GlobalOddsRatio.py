from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
class GlobalOddsRatio(CategoricalCovStruct):
    """
    Estimate the global odds ratio for a GEE with ordinal or nominal
    data.

    References
    ----------
    PJ Heagerty and S Zeger. "Marginal Regression Models for Clustered
    Ordinal Measurements". Journal of the American Statistical
    Association Vol. 91, Issue 435 (1996).

    Thomas Lumley. Generalized Estimating Equations for Ordinal Data:
    A Note on Working Correlation Structures. Biometrics Vol. 52,
    No. 1 (Mar., 1996), pp. 354-361
    http://www.jstor.org/stable/2533173

    Notes
    -----
    The following data structures are calculated in the class:

    'ibd' is a list whose i^th element ibd[i] is a sequence of integer
    pairs (a,b), where endog_li[i][a:b] is the subvector of binary
    indicators derived from the same ordinal value.

    `cpp` is a dictionary where cpp[group] is a map from cut-point
    pairs (c,c') to the indices of all between-subject pairs derived
    from the given cut points.
    """

    def __init__(self, endog_type):
        super().__init__()
        self.endog_type = endog_type
        self.dep_params = 0.0

    def initialize(self, model):
        super().initialize(model)
        if self.model.weights is not None:
            warnings.warn('weights not implemented for GlobalOddsRatio cov_struct, using unweighted covariance estimate', NotImplementedWarning)
        cpp = []
        for v in model.endog_li:
            m = int(len(v) / self._ncut)
            i1, i2 = np.tril_indices(m, -1)
            cpp1 = {}
            for k1 in range(self._ncut):
                for k2 in range(k1 + 1):
                    jj = np.zeros((len(i1), 2), dtype=np.int64)
                    jj[:, 0] = i1 * self._ncut + k1
                    jj[:, 1] = i2 * self._ncut + k2
                    cpp1[k2, k1] = jj
            cpp.append(cpp1)
        self.cpp = cpp
        self.crude_or = self.observed_crude_oddsratio()
        if self.model.update_dep:
            self.dep_params = self.crude_or

    def pooled_odds_ratio(self, tables):
        """
        Returns the pooled odds ratio for a list of 2x2 tables.

        The pooled odds ratio is the inverse variance weighted average
        of the sample odds ratios of the tables.
        """
        if len(tables) == 0:
            return 1.0
        log_oddsratio, var = ([], [])
        for table in tables:
            lor = np.log(table[1, 1]) + np.log(table[0, 0]) - np.log(table[0, 1]) - np.log(table[1, 0])
            log_oddsratio.append(lor)
            var.append((1 / table.astype(np.float64)).sum())
        wts = [1 / v for v in var]
        wtsum = sum(wts)
        wts = [w / wtsum for w in wts]
        log_pooled_or = sum([w * e for w, e in zip(wts, log_oddsratio)])
        return np.exp(log_pooled_or)

    @Appender(CovStruct.covariance_matrix.__doc__)
    def covariance_matrix(self, expected_value, index):
        vmat = self.get_eyy(expected_value, index)
        vmat -= np.outer(expected_value, expected_value)
        return (vmat, False)

    def observed_crude_oddsratio(self):
        """
        To obtain the crude (global) odds ratio, first pool all binary
        indicators corresponding to a given pair of cut points (c,c'),
        then calculate the odds ratio for this 2x2 table.  The crude
        odds ratio is the inverse variance weighted average of these
        odds ratios.  Since the covariate effects are ignored, this OR
        will generally be greater than the stratified OR.
        """
        cpp = self.cpp
        endog = self.model.endog_li
        tables = {}
        for ii in cpp[0].keys():
            tables[ii] = np.zeros((2, 2), dtype=np.float64)
        for i in range(len(endog)):
            yvec = endog[i]
            endog_11 = np.outer(yvec, yvec)
            endog_10 = np.outer(yvec, 1.0 - yvec)
            endog_01 = np.outer(1.0 - yvec, yvec)
            endog_00 = np.outer(1.0 - yvec, 1.0 - yvec)
            cpp1 = cpp[i]
            for ky in cpp1.keys():
                ix = cpp1[ky]
                tables[ky][1, 1] += endog_11[ix[:, 0], ix[:, 1]].sum()
                tables[ky][1, 0] += endog_10[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 1] += endog_01[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 0] += endog_00[ix[:, 0], ix[:, 1]].sum()
        return self.pooled_odds_ratio(list(tables.values()))

    def get_eyy(self, endog_expval, index):
        """
        Returns a matrix V such that V[i,j] is the joint probability
        that endog[i] = 1 and endog[j] = 1, based on the marginal
        probabilities of endog and the global odds ratio `current_or`.
        """
        current_or = self.dep_params
        ibd = self.ibd[index]
        if current_or == 1.0:
            vmat = np.outer(endog_expval, endog_expval)
        else:
            psum = endog_expval[:, None] + endog_expval[None, :]
            pprod = endog_expval[:, None] * endog_expval[None, :]
            pfac = np.sqrt((1.0 + psum * (current_or - 1.0)) ** 2 + 4 * current_or * (1.0 - current_or) * pprod)
            vmat = 1.0 + psum * (current_or - 1.0) - pfac
            vmat /= 2.0 * (current_or - 1)
        for bdl in ibd:
            evy = endog_expval[bdl[0]:bdl[1]]
            if self.endog_type == 'ordinal':
                vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] = np.minimum.outer(evy, evy)
            else:
                vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] = np.diag(evy)
        return vmat

    @Appender(CovStruct.update.__doc__)
    def update(self, params):
        """
        Update the global odds ratio based on the current value of
        params.
        """
        cpp = self.cpp
        cached_means = self.model.cached_means
        if len(cpp[0]) == 0:
            return
        tables = {}
        for ii in cpp[0]:
            tables[ii] = np.zeros((2, 2), dtype=np.float64)
        for i in range(self.model.num_group):
            endog_expval, _ = cached_means[i]
            emat_11 = self.get_eyy(endog_expval, i)
            emat_10 = endog_expval[:, None] - emat_11
            emat_01 = -emat_11 + endog_expval
            emat_00 = 1.0 - (emat_11 + emat_10 + emat_01)
            cpp1 = cpp[i]
            for ky in cpp1.keys():
                ix = cpp1[ky]
                tables[ky][1, 1] += emat_11[ix[:, 0], ix[:, 1]].sum()
                tables[ky][1, 0] += emat_10[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 1] += emat_01[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 0] += emat_00[ix[:, 0], ix[:, 1]].sum()
        cor_expval = self.pooled_odds_ratio(list(tables.values()))
        self.dep_params *= self.crude_or / cor_expval
        if not np.isfinite(self.dep_params):
            self.dep_params = 1.0
            warnings.warn('dep_params became inf, resetting to 1', ConvergenceWarning)

    def summary(self):
        return 'Global odds ratio: %.3f\n' % self.dep_params