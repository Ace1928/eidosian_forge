import numpy as np
from numpy.testing import (assert_array_equal,
from pytest import raises as assert_raises
from scipy.special import gammaln, multigammaln
def _check_multigammaln_array_result(a, d):
    result = multigammaln(a, d)
    assert_array_equal(a.shape, result.shape)
    a1 = a.ravel()
    result1 = result.ravel()
    for i in range(a.size):
        assert_array_almost_equal_nulp(result1[i], multigammaln(a1[i], d))