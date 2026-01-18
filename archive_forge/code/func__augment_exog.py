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
def _augment_exog(self, group_ix):
    """
        Concatenate the columns for variance components to the columns
        for other random effects to obtain a single random effects
        exog matrix for a given group.
        """
    ex_r = self.exog_re_li[group_ix] if self.k_re > 0 else None
    if self.k_vc == 0:
        return ex_r
    ex = [ex_r] if self.k_re > 0 else []
    any_sparse = False
    for j, _ in enumerate(self.exog_vc.names):
        ex.append(self.exog_vc.mats[j][group_ix])
        any_sparse |= sparse.issparse(ex[-1])
    if any_sparse:
        for j, x in enumerate(ex):
            if not sparse.issparse(x):
                ex[j] = sparse.csr_matrix(x)
        ex = sparse.hstack(ex)
        ex = sparse.csr_matrix(ex)
    else:
        ex = np.concatenate(ex, axis=1)
    return ex