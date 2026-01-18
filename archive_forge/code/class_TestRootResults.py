import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
class TestRootResults:
    r = zeros.RootResults(root=1.0, iterations=44, function_calls=46, flag=0, method='newton')

    def test_repr(self):
        expected_repr = '      converged: True\n           flag: converged\n function_calls: 46\n     iterations: 44\n           root: 1.0\n         method: newton'
        assert_equal(repr(self.r), expected_repr)

    def test_type(self):
        assert isinstance(self.r, OptimizeResult)