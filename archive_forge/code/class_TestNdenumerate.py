import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
class TestNdenumerate:

    def test_basic(self):
        a = np.array([[1, 2], [3, 4]])
        assert_equal(list(ndenumerate(a)), [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)])