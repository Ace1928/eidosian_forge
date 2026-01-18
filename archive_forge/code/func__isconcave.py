from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _isconcave(self):
    if not self._cvxterms:
        return True
    else:
        return False