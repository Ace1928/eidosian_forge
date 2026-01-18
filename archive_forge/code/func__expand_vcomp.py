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
def _expand_vcomp(self, vcomp, group_ix):
    """
        Replicate variance parameters to match a group's design.

        Parameters
        ----------
        vcomp : array_like
            The variance parameters for the variance components.
        group_ix : int
            The group index

        Returns an expanded version of vcomp, in which each variance
        parameter is copied as many times as there are independent
        realizations of the variance component in the given group.
        """
    if len(vcomp) == 0:
        return np.empty(0)
    vc_var = []
    for j in range(len(self.exog_vc.names)):
        d = self.exog_vc.mats[j][group_ix].shape[1]
        vc_var.append(vcomp[j] * np.ones(d))
    if len(vc_var) > 0:
        return np.concatenate(vc_var)
    else:
        return np.empty(0)