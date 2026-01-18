import numpy as np
from numpy.linalg import norm
from scipy.linalg import get_blas_funcs
from warnings import warn
def _update_inverse_hessian(self, ys, Hy, yHy, s):
    """Update the inverse Hessian matrix.

        BFGS update using the formula:

            ``H <- H + ((H*y).T*y + s.T*y)/(s.T*y)^2 * (s*s.T)
                     - 1/(s.T*y) * ((H*y)*s.T + s*(H*y).T)``

        where ``s = delta_x`` and ``y = delta_grad``. This formula is
        equivalent to (6.17) in [1]_ written in a more efficient way
        for implementation.

        References
        ----------
        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        """
    self.H = self._syr2(-1.0 / ys, s, Hy, a=self.H)
    self.H = self._syr((ys + yHy) / ys ** 2, s, a=self.H)