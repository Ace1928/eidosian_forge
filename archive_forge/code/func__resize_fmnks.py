import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def _resize_fmnks(self, m, n, k):
    """If necessary, expand the array that remembers PMF values"""
    shape_old = np.array(self._fmnks.shape)
    shape_new = np.array((m + 1, n + 1, k + 1))
    if np.any(shape_new > shape_old):
        shape = np.maximum(shape_old, shape_new)
        fmnks = -np.ones(shape)
        m0, n0, k0 = shape_old
        fmnks[:m0, :n0, :k0] = self._fmnks
        self._fmnks = fmnks