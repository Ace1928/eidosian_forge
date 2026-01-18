from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _splines_x(self, x=None):
    if x is None:
        x = self.x
    n_columns = len(self.knots) + 2
    nobs = x.shape[0]
    basis = np.ones(shape=(nobs, n_columns))
    basis[:, 1] = x
    for i, xi in enumerate(x):
        for j, xkj in enumerate(self.knots):
            s_ij = self._rk(xi, xkj)
            basis[i, j + 2] = s_ij
    return basis