import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestMayShareMemory:

    def test_basic(self):
        d = np.ones((50, 60))
        d2 = np.ones((30, 60, 6))
        assert_(np.may_share_memory(d, d))
        assert_(np.may_share_memory(d, d[::-1]))
        assert_(np.may_share_memory(d, d[::2]))
        assert_(np.may_share_memory(d, d[1:, ::-1]))
        assert_(not np.may_share_memory(d[::-1], d2))
        assert_(not np.may_share_memory(d[::2], d2))
        assert_(not np.may_share_memory(d[1:, ::-1], d2))
        assert_(np.may_share_memory(d2[1:, ::-1], d2))