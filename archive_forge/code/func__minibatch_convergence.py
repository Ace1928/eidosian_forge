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
def _minibatch_convergence(self, X, batch_cost, H, H_buffer, n_samples, step, n_steps):
    """Helper function to encapsulate the early stopping logic"""
    batch_size = X.shape[0]
    step = step + 1
    if step == 1:
        if self.verbose:
            print(f'Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}')
        return False
    if self._ewa_cost is None:
        self._ewa_cost = batch_cost
    else:
        alpha = batch_size / (n_samples + 1)
        alpha = min(alpha, 1)
        self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha
    if self.verbose:
        print(f'Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}, ewa cost: {self._ewa_cost}')
    H_diff = linalg.norm(H - H_buffer) / linalg.norm(H)
    if self.tol > 0 and H_diff <= self.tol:
        if self.verbose:
            print(f'Converged (small H change) at step {step}/{n_steps}')
        return True
    if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
        self._no_improvement = 0
        self._ewa_cost_min = self._ewa_cost
    else:
        self._no_improvement += 1
    if self.max_no_improvement is not None and self._no_improvement >= self.max_no_improvement:
        if self.verbose:
            print(f'Converged (lack of improvement in objective function) at step {step}/{n_steps}')
        return True
    return False