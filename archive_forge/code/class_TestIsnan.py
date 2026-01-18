import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestIsnan:

    def test_goodvalues(self):
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isnan(z) == 0
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        with np.errstate(divide='ignore'):
            assert_all(np.isnan(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        with np.errstate(divide='ignore'):
            assert_all(np.isnan(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isnan(np.array((0.0,)) / 0.0) == 1)

    def test_integer(self):
        assert_all(np.isnan(1) == 0)

    def test_complex(self):
        assert_all(np.isnan(1 + 1j) == 0)

    def test_complex1(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isnan(np.array(0 + 0j) / 0.0) == 1)