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
class TestLinprogIPSparsePresolve(LinprogIPTests):
    options = {'sparse': True, '_sparse_presolve': True}

    @pytest.mark.xfail_on_32bit('This test is sensitive to machine epsilon level perturbations in linear system solution in _linprog_ip._sym_solve.')
    def test_bug_6139(self):
        super().test_bug_6139()

    def test_enzo_example_c_with_infeasibility(self):
        pytest.skip('_sparse_presolve=True incompatible with presolve=False')

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        super().test_bug_6690()