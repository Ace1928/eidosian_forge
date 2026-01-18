from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _equally_spaced_knots(x, df):
    n_knots = df - 2
    x_min = x.min()
    x_max = x.max()
    knots = np.linspace(x_min, x_max, n_knots)
    return knots