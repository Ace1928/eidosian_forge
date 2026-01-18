from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _isconvex(self):
    if not self._ccvterms:
        return True
    else:
        return False