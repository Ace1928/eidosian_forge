import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
class Test_measurements_select:
    """ndimage._measurements._select() is a utility used by other functions."""

    def test_basic(self):
        x = [0, 1, 6, 2]
        cases = [([0, 0, 1, 1], [0, 1]), ([0, 0, 9, 9], [0, 9]), ([0.0, 0.0, 7.0, 7.0], [0.0, 7.0])]
        for labels, index in cases:
            result = ndimage._measurements._select(x, labels=labels, index=index)
            assert_(len(result) == 0)
            result = ndimage._measurements._select(x, labels=labels, index=index, find_max=True)
            assert_(len(result) == 1)
            assert_array_equal(result[0], [1, 6])
            result = ndimage._measurements._select(x, labels=labels, index=index, find_min=True)
            assert_(len(result) == 1)
            assert_array_equal(result[0], [0, 2])
            result = ndimage._measurements._select(x, labels=labels, index=index, find_min=True, find_min_positions=True)
            assert_(len(result) == 2)
            assert_array_equal(result[0], [0, 2])
            assert_array_equal(result[1], [0, 3])
            assert_equal(result[1].dtype.kind, 'i')
            result = ndimage._measurements._select(x, labels=labels, index=index, find_max=True, find_max_positions=True)
            assert_(len(result) == 2)
            assert_array_equal(result[0], [1, 6])
            assert_array_equal(result[1], [1, 2])
            assert_equal(result[1].dtype.kind, 'i')