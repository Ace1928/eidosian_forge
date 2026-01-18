import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestApplyOverAxes:

    def test_simple(self):
        a = np.arange(24).reshape(2, 3, 4)
        aoa_a = apply_over_axes(np.sum, a, [0, 2])
        assert_array_equal(aoa_a, np.array([[[60], [92], [124]]]))