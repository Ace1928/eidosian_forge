import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _ensure_constraint(self, trial):
    """Make sure the parameters lie between the limits."""
    mask = np.where((trial > 1) | (trial < 0))
    trial[mask] = self.random_number_generator.uniform(size=mask[0].shape)