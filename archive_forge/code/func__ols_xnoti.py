import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
def _ols_xnoti(self, drop_idx, endog_idx='endog', store=True):
    """regression results from LOVO auxiliary regression with cache


        The result instances are stored, which could use a large amount of
        memory if the datasets are large. There are too many combinations to
        store them all, except for small problems.

        Parameters
        ----------
        drop_idx : int
            index of exog that is dropped from the regression
        endog_idx : 'endog' or int
            If 'endog', then the endogenous variable of the result instance
            is regressed on the exogenous variables, excluding the one at
            drop_idx. If endog_idx is an integer, then the exog with that
            index is regressed with OLS on all other exogenous variables.
            (The latter is the auxiliary regression for the variance inflation
            factor.)

        this needs more thought, memory versus speed
        not yet used in any other parts, not sufficiently tested
        """
    if endog_idx == 'endog':
        stored = self.aux_regression_endog
        if hasattr(stored, drop_idx):
            return stored[drop_idx]
        x_i = self.results.model.endog
    else:
        try:
            self.aux_regression_exog[endog_idx][drop_idx]
        except KeyError:
            pass
        stored = self.aux_regression_exog[endog_idx]
        stored = {}
        x_i = self.exog[:, endog_idx]
    k_vars = self.exog.shape[1]
    mask = np.arange(k_vars) != drop_idx
    x_noti = self.exog[:, mask]
    res = OLS(x_i, x_noti).fit()
    if store:
        stored[drop_idx] = res
    return res