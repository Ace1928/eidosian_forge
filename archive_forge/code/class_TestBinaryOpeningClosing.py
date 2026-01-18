import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
class TestBinaryOpeningClosing:

    def setup_method(self):
        a = numpy.zeros((5, 5), dtype=bool)
        a[1:4, 1:4] = True
        a[4, 4] = True
        self.array = a
        self.sq3x3 = numpy.ones((3, 3))
        self.opened_old = ndimage.binary_opening(self.array, self.sq3x3, 1, None, 0)
        self.closed_old = ndimage.binary_closing(self.array, self.sq3x3, 1, None, 0)

    def test_opening_new_arguments(self):
        opened_new = ndimage.binary_opening(self.array, self.sq3x3, 1, None, 0, None, 0, False)
        assert_array_equal(opened_new, self.opened_old)

    def test_closing_new_arguments(self):
        closed_new = ndimage.binary_closing(self.array, self.sq3x3, 1, None, 0, None, 0, False)
        assert_array_equal(closed_new, self.closed_old)