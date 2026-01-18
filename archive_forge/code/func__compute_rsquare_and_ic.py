import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _compute_rsquare_and_ic(self):
    """
        Final statistics to compute
        """
    weights = self.weights
    ss_data = self.transformed_data * np.sqrt(weights)
    self._tss_indiv = np.sum(ss_data ** 2, 0)
    self._tss = np.sum(self._tss_indiv)
    self._ess = np.zeros(self._ncomp + 1)
    self._ess_indiv = np.zeros((self._ncomp + 1, self._nvar))
    for i in range(self._ncomp + 1):
        projection = self.project(ncomp=i, transform=False, unweight=False)
        indiv_rss = (projection ** 2).sum(axis=0)
        rss = indiv_rss.sum()
        self._ess[i] = self._tss - rss
        self._ess_indiv[i, :] = self._tss_indiv - indiv_rss
    self.rsquare = 1.0 - self._ess / self._tss
    ess = self._ess
    invalid = ess <= 0
    if invalid.any():
        last_obs = np.where(invalid)[0].min()
        ess = ess[:last_obs]
    log_ess = np.log(ess)
    r = np.arange(ess.shape[0])
    nobs, nvar = (self._nobs, self._nvar)
    sum_to_prod = (nobs + nvar) / (nobs * nvar)
    min_dim = min(nobs, nvar)
    penalties = np.array([sum_to_prod * np.log(1.0 / sum_to_prod), sum_to_prod * np.log(min_dim), np.log(min_dim) / min_dim])
    penalties = penalties[:, None]
    ic = log_ess + r * penalties
    self.ic = ic.T