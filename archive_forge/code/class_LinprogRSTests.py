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
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class LinprogRSTests(LinprogCommonTests):
    method = 'revised simplex'

    def test_bug_5400(self):
        pytest.skip('Intermittent failure acceptable.')

    def test_bug_8662(self):
        pytest.skip('Intermittent failure acceptable.')

    def test_network_flow(self):
        pytest.skip('Intermittent failure acceptable.')