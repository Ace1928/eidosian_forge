from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class TestFlipud:

    def test_basic(self):
        a = get_mat(4)
        b = a[::-1, :]
        assert_equal(flipud(a), b)
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[3, 4, 5], [0, 1, 2]]
        assert_equal(flipud(a), b)