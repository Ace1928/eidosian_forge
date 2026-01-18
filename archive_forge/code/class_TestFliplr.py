from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class TestFliplr:

    def test_basic(self):
        assert_raises(ValueError, fliplr, ones(4))
        a = get_mat(4)
        b = a[:, ::-1]
        assert_equal(fliplr(a), b)
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[2, 1, 0], [5, 4, 3]]
        assert_equal(fliplr(a), b)