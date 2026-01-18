import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
class TestErrorChecking:

    def test_option_lsmr_tol(self):
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=0.01)
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol='auto')
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=None)
        err_message = "`lsmr_tol` must be None, 'auto', or positive float."
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=-0.1)
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol='foo')
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_tol=1)

    def test_option_lsmr_maxiter(self):
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=1)
        _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=None)
        err_message = '`lsmr_maxiter` must be None or positive integer.'
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=0)
        with pytest.raises(ValueError, match=err_message):
            _ = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=-1)