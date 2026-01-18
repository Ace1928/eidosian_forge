from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _get_integration_points(knots, k_points=3):
    """add points to each subinterval defined by knots

    inserts k_points between each two consecutive knots
    """
    k_points = k_points + 1
    knots = np.unique(knots)
    dxi = np.arange(k_points) / k_points
    dxk = np.diff(knots)
    dx = dxk[:, None] * dxi
    x = np.concatenate(((knots[:-1, None] + dx).ravel(), [knots[-1]]))
    return x