import itertools
from numbers import Integral, Real
import numpy as np
from scipy.special import gammainc
from ..base import BaseEstimator, _fit_context
from ..neighbors._base import VALID_METRICS
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._ball_tree import BallTree
from ._kd_tree import KDTree
def _choose_algorithm(self, algorithm, metric):
    if algorithm == 'auto':
        if metric in KDTree.valid_metrics:
            return 'kd_tree'
        elif metric in BallTree.valid_metrics:
            return 'ball_tree'
    else:
        if metric not in TREE_DICT[algorithm].valid_metrics:
            raise ValueError("invalid metric for {0}: '{1}'".format(TREE_DICT[algorithm], metric))
        return algorithm