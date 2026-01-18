import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
class TestChar:

    def setup_method(self):
        self.A = np.array('abc1', dtype='c').view(np.chararray)

    def test_it(self):
        assert_equal(self.A.shape, (4,))
        assert_equal(self.A.upper()[:2].tobytes(), b'AB')