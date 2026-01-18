from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _islp(self):
    """ 
        Returns True if self is an LP; False otherwise.
        """
    if not self.objective._isaffine():
        return False
    for c in self._inequalities:
        if not c._f._isaffine():
            return False
    for c in self._equalities:
        if not c._f._isaffine():
            return False
    return True