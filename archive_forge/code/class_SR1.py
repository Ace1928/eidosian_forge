import numpy as np
from numpy.linalg import norm
from scipy.linalg import get_blas_funcs
from warnings import warn
class SR1(FullHessianUpdateStrategy):
    """Symmetric-rank-1 Hessian update strategy.

    Parameters
    ----------
    min_denominator : float
        This number, scaled by a normalization factor,
        defines the minimum denominator magnitude allowed
        in the update. When the condition is violated we skip
        the update. By default uses ``1e-8``.
    init_scale : {float, 'auto'}, optional
        Matrix scale at first iteration. At the first
        iteration the Hessian matrix or its inverse will be initialized
        with ``init_scale*np.eye(n)``, where ``n`` is the problem dimension.
        Set it to 'auto' in order to use an automatic heuristic for choosing
        the initial scale. The heuristic is described in [1]_, p.143.
        By default uses 'auto'.

    Notes
    -----
    The update is based on the description in [1]_, p.144-146.

    References
    ----------
    .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """

    def __init__(self, min_denominator=1e-08, init_scale='auto'):
        self.min_denominator = min_denominator
        super().__init__(init_scale)

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