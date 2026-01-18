from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class TestTrilIndicesFrom:

    def test_exceptions(self):
        assert_raises(ValueError, tril_indices_from, np.ones((2,)))
        assert_raises(ValueError, tril_indices_from, np.ones((2, 2, 2)))