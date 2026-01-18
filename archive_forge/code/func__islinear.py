from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _islinear(self):
    if len(self._constant) == 1 and (not self._constant[0]) and (not self._cvxterms) and (not self._ccvterms):
        return True
    else:
        return False