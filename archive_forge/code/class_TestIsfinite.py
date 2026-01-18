import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestIsfinite:

    def test_goodvalues(self):
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isfinite(z) == 1
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isfinite(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isfinite(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isfinite(np.array((0.0,)) / 0.0) == 0)

    def test_integer(self):
        assert_all(np.isfinite(1) == 1)

    def test_complex(self):
        assert_all(np.isfinite(1 + 1j) == 1)

    def test_complex1(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_all(np.isfinite(np.array(1 + 1j) / 0.0) == 0)