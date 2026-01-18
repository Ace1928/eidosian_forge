import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestIsinf:

    def test_goodvalues(self):
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isinf(z) == 0
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isinf(np.array((1.0,)) / 0.0) == 1)

    def test_posinf_scalar(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isinf(np.array(1.0) / 0.0) == 1)

    def test_neginf(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isinf(np.array((-1.0,)) / 0.0) == 1)

    def test_neginf_scalar(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isinf(np.array(-1.0) / 0.0) == 1)

    def test_ind(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isinf(np.array((0.0,)) / 0.0) == 0)