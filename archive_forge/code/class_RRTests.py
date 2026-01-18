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
class RRTests:
    method = 'interior-point'
    LCT = LinprogCommonTests
    test_RR_infeasibility = LCT.test_remove_redundancy_infeasibility
    test_bug_10349 = LCT.test_bug_10349
    test_bug_7044 = LCT.test_bug_7044
    test_NFLC = LCT.test_network_flow_limited_capacity
    test_enzo_example_b = LCT.test_enzo_example_b