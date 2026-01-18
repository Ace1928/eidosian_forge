import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
class TestDiagIndicesFrom:

    def test_diag_indices_from(self):
        x = np.random.random((4, 4))
        r, c = diag_indices_from(x)
        assert_array_equal(r, np.arange(4))
        assert_array_equal(c, np.arange(4))

    def test_error_small_input(self):
        x = np.ones(7)
        with assert_raises_regex(ValueError, 'at least 2-d'):
            diag_indices_from(x)

    def test_error_shape_mismatch(self):
        x = np.zeros((3, 3, 2, 3), int)
        with assert_raises_regex(ValueError, 'equal length'):
            diag_indices_from(x)