import numpy as np
from .base import OdeSolver, DenseOutput
from .common import (validate_max_step, validate_tol, select_initial_step,
from . import dop853_coefficients
def _dense_output_impl(self):
    K = self.K_extended
    h = self.h_previous
    for s, (a, c) in enumerate(zip(self.A_EXTRA, self.C_EXTRA), start=self.n_stages + 1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = self.fun(self.t_old + c * h, self.y_old + dy)
    F = np.empty((dop853_coefficients.INTERPOLATOR_POWER, self.n), dtype=self.y_old.dtype)
    f_old = K[0]
    delta_y = self.y - self.y_old
    F[0] = delta_y
    F[1] = h * f_old - delta_y
    F[2] = 2 * delta_y - h * (self.f + f_old)
    F[3:] = h * np.dot(self.D, K)
    return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)