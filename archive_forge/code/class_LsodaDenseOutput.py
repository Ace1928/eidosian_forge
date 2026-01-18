import numpy as np
from scipy.integrate import ode
from .common import validate_tol, validate_first_step, warn_extraneous
from .base import OdeSolver, DenseOutput
class LsodaDenseOutput(DenseOutput):

    def __init__(self, t_old, t, h, order, yh):
        super().__init__(t_old, t)
        self.h = h
        self.yh = yh
        self.p = np.arange(order + 1)

    def _call_impl(self, t):
        if t.ndim == 0:
            x = ((t - self.t) / self.h) ** self.p
        else:
            x = ((t - self.t) / self.h) ** self.p[:, None]
        return np.dot(self.yh, x)