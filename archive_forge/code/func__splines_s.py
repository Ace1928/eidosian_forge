from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _splines_s(self):
    q = len(self.knots) + 2
    s = np.zeros(shape=(q, q))
    for i, x1 in enumerate(self.knots):
        for j, x2 in enumerate(self.knots):
            s[i + 2, j + 2] = self._rk(x1, x2)
    return s