from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _rk(self, x, z):
    p1 = ((z - 1 / 2) ** 2 - 1 / 12) * ((x - 1 / 2) ** 2 - 1 / 12) / 4
    p2 = ((np.abs(z - x) - 1 / 2) ** 4 - 1 / 2 * (np.abs(z - x) - 1 / 2) ** 2 + 7 / 240) / 24.0
    return p1 - p2