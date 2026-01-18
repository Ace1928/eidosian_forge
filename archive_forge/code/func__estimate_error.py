import numpy as np
from .base import OdeSolver, DenseOutput
from .common import (validate_max_step, validate_tol, select_initial_step,
from . import dop853_coefficients
def _estimate_error(self, K, h):
    err5 = np.dot(K.T, self.E5)
    err3 = np.dot(K.T, self.E3)
    denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
    correction_factor = np.ones_like(err5)
    mask = denom > 0
    correction_factor[mask] = np.abs(err5[mask]) / denom[mask]
    return h * err5 * correction_factor