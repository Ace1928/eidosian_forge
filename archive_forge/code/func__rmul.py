from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _rmul(self, a):
    """ 
        self := a*self where a is scalar or matrix 
        """
    lg = len(self)
    if _isscalar(a):
        for v in iter(self._coeff.keys()):
            self._coeff[v] *= a
    elif lg == 1 and _ismatrix(a) and (a.size[1] == 1):
        for v, c in iter(self._coeff.items()):
            self._coeff[v] = a * c
    elif _ismatrix(a) and a.size[1] == lg:
        for v, c in iter(self._coeff.items()):
            if c.size == (1, len(v)):
                self._coeff[v] = a * c[lg * [0], :]
            else:
                self._coeff[v] = a * c
    else:
        raise TypeError('incompatible dimensions')