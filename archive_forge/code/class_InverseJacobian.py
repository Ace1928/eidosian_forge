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
class InverseJacobian:

    def __init__(self, jacobian):
        self.jacobian = jacobian
        self.matvec = jacobian.solve
        self.update = jacobian.update
        if hasattr(jacobian, 'setup'):
            self.setup = jacobian.setup
        if hasattr(jacobian, 'rsolve'):
            self.rmatvec = jacobian.rsolve

    @property
    def shape(self):
        return self.jacobian.shape

    @property
    def dtype(self):
        return self.jacobian.dtype