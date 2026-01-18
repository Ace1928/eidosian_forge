import numpy as np
from numpy.testing import (
from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id
from scipy.sparse import csc_matrix
class TestRRPivotSparse(RRCommonTests):

    def rr(self, A, b):
        rr_res = _remove_redundancy_pivot_sparse(csc_matrix(A), b)
        A1, b1, status, message = rr_res
        return (A1.toarray(), b1, status, message)