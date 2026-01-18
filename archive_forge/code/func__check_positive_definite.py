import math
import warnings
import sys
import inspect
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
def _check_positive_definite(Hk):

    def is_pos_def(A):
        if issymmetric(A):
            try:
                cholesky(A)
                return True
            except LinAlgError:
                return False
        else:
            return False
    if Hk is not None:
        if not is_pos_def(Hk):
            raise ValueError("'hess_inv0' matrix isn't positive definite.")