import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
class GenericBroyden(Jacobian):

    def setup(self, x0, f0, func):
        Jacobian.setup(self, x0, f0, func)
        self.last_f = f0
        self.last_x = x0
        if hasattr(self, 'alpha') and self.alpha is None:
            normf0 = norm(f0)
            if normf0:
                self.alpha = 0.5 * max(norm(x0), 1) / normf0
            else:
                self.alpha = 1.0

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        raise NotImplementedError

    def update(self, x, f):
        df = f - self.last_f
        dx = x - self.last_x
        self._update(x, f, dx, df, norm(dx), norm(df))
        self.last_f = f
        self.last_x = x