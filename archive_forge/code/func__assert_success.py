import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def _assert_success(res, desired_fun=None, desired_x=None, rtol=1e-08, atol=1e-08):
    if not res.success:
        msg = f'linprog status {res.status}, message: {res.message}'
        raise AssertionError(msg)
    assert_equal(res.status, 0)
    if desired_fun is not None:
        assert_allclose(res.fun, desired_fun, err_msg='converged to an unexpected objective value', rtol=rtol, atol=atol)
    if desired_x is not None:
        assert_allclose(res.x, desired_x, err_msg='converged to an unexpected solution', rtol=rtol, atol=atol)