import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
class TestDilateFix:

    def setup_method(self):
        self.array = numpy.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=numpy.uint8)
        self.sq3x3 = numpy.ones((3, 3))
        dilated3x3 = ndimage.binary_dilation(self.array, structure=self.sq3x3)
        self.dilated3x3 = dilated3x3.view(numpy.uint8)

    def test_dilation_square_structure(self):
        result = ndimage.grey_dilation(self.array, structure=self.sq3x3)
        assert_array_almost_equal(result, self.dilated3x3 + 1)

    def test_dilation_scalar_size(self):
        result = ndimage.grey_dilation(self.array, size=3)
        assert_array_almost_equal(result, self.dilated3x3)