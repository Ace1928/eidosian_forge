import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
class TestWatershedIft:

    def test_watershed_ift01(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        expected = [[-1, -1, -1, -1, -1, -1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift02(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers)
        expected = [[-1, -1, -1, -1, -1, -1, -1], [-1, -1, 1, 1, 1, -1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift03(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = ndimage.watershed_ift(data, markers)
        expected = [[-1, -1, -1, -1, -1, -1, -1], [-1, -1, 2, -1, 3, -1, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, -1, 2, -1, 3, -1, -1], [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift04(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = ndimage.watershed_ift(data, markers, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        expected = [[-1, -1, -1, -1, -1, -1, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift05(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = ndimage.watershed_ift(data, markers, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        expected = [[-1, -1, -1, -1, -1, -1, -1], [-1, 3, 3, 2, 2, 2, -1], [-1, 3, 3, 2, 2, 2, -1], [-1, 3, 3, 2, 2, 2, -1], [-1, 3, 3, 2, 2, 2, -1], [-1, 3, 3, 2, 2, 2, -1], [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift06(self):
        data = np.array([[0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        expected = [[-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift07(self):
        shape = (7, 6)
        data = np.zeros(shape, dtype=np.uint8)
        data = data.transpose()
        data[...] = np.array([[0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = np.zeros(shape, dtype=np.int16)
        out = out.transpose()
        ndimage.watershed_ift(data, markers, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], output=out)
        expected = [[-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift08(self):
        data = np.array([[256, 0], [0, 0]], np.uint16)
        markers = np.array([[1, 0], [0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers)
        expected = [[1, 1], [1, 1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift09(self):
        data = np.array([[np.iinfo(np.uint16).max, 0], [0, 0]], np.uint16)
        markers = np.array([[1, 0], [0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers)
        expected = [[1, 1], [1, 1]]
        assert_allclose(out, expected)