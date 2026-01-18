import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestEmptyField:

    def test_assign(self):
        a = np.arange(10, dtype=np.float32)
        a.dtype = [('int', '<0i4'), ('float', '<2f4')]
        assert_(a['int'].shape == (5, 0))
        assert_(a['float'].shape == (5, 2))