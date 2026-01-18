import numpy as np
from .base import OdeSolver, DenseOutput
from .common import (validate_max_step, validate_tol, select_initial_step,
from . import dop853_coefficients
def _step_impl(self):
    t = self.t
    y = self.y
    max_step = self.max_step
    rtol = self.rtol
    atol = self.atol
    min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
    if self.h_abs > max_step:
        h_abs = max_step
    elif self.h_abs < min_step:
        h_abs = min_step
    else:
        h_abs = self.h_abs
    step_accepted = False
    step_rejected = False
    while not step_accepted:
        if h_abs < min_step:
            return (False, self.TOO_SMALL_STEP)
        h = h_abs * self.direction
        t_new = t + h
        if self.direction * (t_new - self.t_bound) > 0:
            t_new = self.t_bound
        h = t_new - t
        h_abs = np.abs(h)
        y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K)
        scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
        error_norm = self._estimate_error_norm(self.K, h, scale)
        if error_norm < 1:
            if error_norm == 0:
                factor = MAX_FACTOR
            else:
                factor = min(MAX_FACTOR, SAFETY * error_norm ** self.error_exponent)
            if step_rejected:
                factor = min(1, factor)
            h_abs *= factor
            step_accepted = True
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** self.error_exponent)
            step_rejected = True
    self.h_previous = h
    self.y_old = y
    self.t = t_new
    self.y = y_new
    self.h_abs = h_abs
    self.f = f_new
    return (True, None)