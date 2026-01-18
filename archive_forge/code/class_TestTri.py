from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class TestTri:

    def test_dtype(self):
        out = array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        assert_array_equal(tri(3), out)
        assert_array_equal(tri(3, dtype=bool), out.astype(bool))