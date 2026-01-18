from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _isspmatrix(a):
    """ True if a is a nonempty sparse 'd' matrix. """
    if type(a) is spmatrix and a.typecode == 'd' and (min(a.size) != 0):
        return True
    else:
        return False