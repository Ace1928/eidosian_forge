import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestIscomplex:

    def test_fail(self):
        z = np.array([-1, 0, 1])
        res = iscomplex(z)
        assert_(not np.any(res, axis=0))

    def test_pass(self):
        z = np.array([-1j, 1, 0])
        res = iscomplex(z)
        assert_array_equal(res, [1, 0, 0])