import numpy as np
from numpy.linalg import norm
from scipy.linalg import get_blas_funcs
from warnings import warn
def _update_implementation(self, delta_x, delta_grad):
    if self.approx_type == 'hess':
        w = delta_x
        z = delta_grad
    else:
        w = delta_grad
        z = delta_x
    Mw = self.dot(w)
    z_minus_Mw = z - Mw
    denominator = np.dot(w, z_minus_Mw)
    if np.abs(denominator) <= self.min_denominator * norm(w) * norm(z_minus_Mw):
        return
    if self.approx_type == 'hess':
        self.B = self._syr(1 / denominator, z_minus_Mw, a=self.B)
    else:
        self.H = self._syr(1 / denominator, z_minus_Mw, a=self.H)