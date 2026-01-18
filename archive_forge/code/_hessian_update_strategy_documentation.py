import numpy as np
from numpy.linalg import norm
from scipy.linalg import get_blas_funcs
from warnings import warn
Update the Hessian matrix.

        BFGS update using the formula:

            ``B <- B - (B*s)*(B*s).T/s.T*(B*s) + y*y^T/s.T*y``

        where ``s`` is short for ``delta_x`` and ``y`` is short
        for ``delta_grad``. Formula (6.19) in [1]_.

        References
        ----------
        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        