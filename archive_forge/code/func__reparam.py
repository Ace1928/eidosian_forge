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
def _reparam(self):
    """
        Returns parameters of the map converting parameters from the
        form used in optimization to the form returned to the user.

        Returns
        -------
        lin : list-like
            Linear terms of the map
        quad : list-like
            Quadratic terms of the map

        Notes
        -----
        If P are the standard form parameters and R are the
        transformed parameters (i.e. with the Cholesky square root
        covariance and square root transformed variance components),
        then P[i] = lin[i] * R + R' * quad[i] * R
        """
    k_fe, k_re, k_re2, k_vc = (self.k_fe, self.k_re, self.k_re2, self.k_vc)
    k_tot = k_fe + k_re2 + k_vc
    ix = np.tril_indices(self.k_re)
    lin = []
    for k in range(k_fe):
        e = np.zeros(k_tot)
        e[k] = 1
        lin.append(e)
    for k in range(k_re2):
        lin.append(np.zeros(k_tot))
    for k in range(k_vc):
        lin.append(np.zeros(k_tot))
    quad = []
    for k in range(k_tot):
        quad.append(np.zeros((k_tot, k_tot)))
    ii = np.tril_indices(k_re)
    ix = [(a, b) for a, b in zip(ii[0], ii[1])]
    for i1 in range(k_re2):
        for i2 in range(k_re2):
            ix1 = ix[i1]
            ix2 = ix[i2]
            if ix1[1] == ix2[1] and ix1[0] <= ix2[0]:
                ii = (ix2[0], ix1[0])
                k = ix.index(ii)
                quad[k_fe + k][k_fe + i2, k_fe + i1] += 1
    for k in range(k_tot):
        quad[k] = 0.5 * (quad[k] + quad[k].T)
    km = k_fe + k_re2
    for k in range(km, km + k_vc):
        quad[k][k, k] = 1
    return (lin, quad)