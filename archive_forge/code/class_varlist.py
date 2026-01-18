from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
class varlist(list):
    """
    Standard list with __contains__() redefined to use 'is' 
    instead of '=='.
    """

    def __contains__(self, item):
        for k in range(len(self)):
            if self[k] is item:
                return True
        return False