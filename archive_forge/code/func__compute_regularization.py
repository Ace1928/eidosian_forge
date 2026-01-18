import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state, gen_batches, metadata_routing
from ..utils._param_validation import (
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import (
from ._cdnmf_fast import _update_cdnmf_fast
def _compute_regularization(self, X):
    """Compute scaled regularization terms."""
    n_samples, n_features = X.shape
    alpha_W = self.alpha_W
    alpha_H = self.alpha_W if self.alpha_H == 'same' else self.alpha_H
    l1_reg_W = n_features * alpha_W * self.l1_ratio
    l1_reg_H = n_samples * alpha_H * self.l1_ratio
    l2_reg_W = n_features * alpha_W * (1.0 - self.l1_ratio)
    l2_reg_H = n_samples * alpha_H * (1.0 - self.l1_ratio)
    return (l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H)