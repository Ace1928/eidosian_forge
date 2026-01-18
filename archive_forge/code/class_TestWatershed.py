import math
import unittest
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage._shared.filters import gaussian
from skimage.measure import label
from .._watershed import watershed
class TestWatershed(unittest.TestCase):
    eight = np.ones((3, 3), bool)

    def test_watershed01(self):
        """watershed 1"""
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = watershed(data, markers, self.eight)
        expected = np.array([[-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]])
        error = diff(expected, out)
        assert error < eps

    def test_watershed02(self):
        """watershed 2"""
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = watershed(data, markers)
        error = diff([[-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, 1, 1, 1, -1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed03(self):
        """watershed 3"""
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = watershed(data, markers)
        error = diff([[-1, -1, -1, -1, -1, -1, -1], [-1, 0, 2, 0, 3, 0, -1], [-1, 2, 2, 0, 3, 3, -1], [-1, 2, 2, 0, 3, 3, -1], [-1, 2, 2, 0, 3, 3, -1], [-1, 0, 2, 0, 3, 0, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed04(self):
        """watershed 4"""
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = watershed(data, markers, self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1], [-1, 2, 2, 0, 3, 3, -1], [-1, 2, 2, 0, 3, 3, -1], [-1, 2, 2, 0, 3, 3, -1], [-1, 2, 2, 0, 3, 3, -1], [-1, 2, 2, 0, 3, 3, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed05(self):
        """watershed 5"""
        data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = watershed(data, markers, self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1], [-1, 3, 3, 0, 2, 2, -1], [-1, 3, 3, 0, 2, 2, -1], [-1, 3, 3, 0, 2, 2, -1], [-1, 3, 3, 0, 2, 2, -1], [-1, 3, 3, 0, 2, 2, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed06(self):
        """watershed 6"""
        data = np.array([[0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0]], np.int8)
        out = watershed(data, markers, self.eight)
        error = diff([[-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed07(self):
        """A regression test of a competitive case that failed"""
        data = blob
        mask = data != 255
        markers = np.zeros(data.shape, int)
        markers[6, 7] = 1
        markers[14, 7] = 2
        out = watershed(data, markers, self.eight, mask=mask)
        size1 = np.sum(out == 1)
        size2 = np.sum(out == 2)
        self.assertTrue(abs(size1 - size2) <= 6)

    def test_watershed08(self):
        """The border pixels + an edge are all the same value"""
        data = blob.copy()
        data[10, 7:9] = 141
        mask = data != 255
        markers = np.zeros(data.shape, int)
        markers[6, 7] = 1
        markers[14, 7] = 2
        out = watershed(data, markers, self.eight, mask=mask)
        size1 = np.sum(out == 1)
        size2 = np.sum(out == 2)
        self.assertTrue(abs(size1 - size2) <= 6)

    def test_watershed09(self):
        """Test on an image of reasonable size

        This is here both for timing (does it take forever?) and to
        ensure that the memory constraints are reasonable
        """
        image = np.zeros((1000, 1000))
        coords = np.random.uniform(0, 1000, (100, 2)).astype(int)
        markers = np.zeros((1000, 1000), int)
        idx = 1
        for x, y in coords:
            image[x, y] = 1
            markers[x, y] = idx
            idx += 1
        image = gaussian(image, sigma=4, mode='reflect')
        watershed(image, markers, self.eight)
        ndi.watershed_ift(image.astype(np.uint16), markers, self.eight)

    def test_watershed10(self):
        """watershed 10"""
        data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], np.uint8)
        markers = np.array([[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]], np.int8)
        out = watershed(data, markers, self.eight)
        error = diff([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], out)
        self.assertTrue(error < eps)

    def test_watershed11(self):
        """Make sure that all points on this plateau are assigned to closest seed"""
        image = np.zeros((21, 21))
        markers = np.zeros((21, 21), int)
        markers[5, 5] = 1
        markers[5, 10] = 2
        markers[10, 5] = 3
        markers[10, 10] = 4
        structure = np.array([[False, True, False], [True, True, True], [False, True, False]])
        out = watershed(image, markers, structure)
        i, j = np.mgrid[0:21, 0:21]
        d = np.dstack([np.sqrt((i.astype(float) - i0) ** 2, (j.astype(float) - j0) ** 2) for i0, j0 in ((5, 5), (5, 10), (10, 5), (10, 10))])
        dmin = np.min(d, 2)
        self.assertTrue(np.all(d[i, j, out[i, j] - 1] == dmin))

    def test_watershed12(self):
        """The watershed line"""
        data = np.array([[203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153], [203, 255, 203, 153, 153, 153, 102, 102, 102, 102, 102, 102, 153, 153, 153, 153], [203, 255, 203, 203, 153, 153, 102, 102, 77, 0, 102, 102, 153, 153, 203, 203], [203, 255, 255, 203, 153, 153, 153, 102, 102, 102, 102, 153, 153, 203, 203, 255], [203, 203, 255, 203, 203, 203, 153, 153, 153, 153, 153, 153, 203, 203, 255, 255], [153, 203, 255, 255, 255, 203, 203, 203, 203, 203, 203, 203, 203, 255, 255, 203], [153, 203, 203, 203, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 203, 203], [153, 153, 153, 203, 203, 203, 203, 203, 255, 203, 203, 203, 203, 203, 203, 153], [102, 102, 153, 153, 153, 153, 203, 203, 255, 203, 203, 255, 203, 153, 153, 153], [102, 102, 102, 102, 102, 153, 203, 255, 255, 203, 203, 203, 203, 153, 102, 153], [102, 51, 51, 102, 102, 153, 203, 255, 203, 203, 153, 153, 153, 153, 102, 153], [77, 51, 51, 102, 153, 153, 203, 255, 203, 203, 203, 153, 102, 102, 102, 153], [77, 0, 51, 102, 153, 203, 203, 255, 203, 255, 203, 153, 102, 51, 102, 153], [77, 0, 51, 102, 153, 203, 255, 255, 203, 203, 203, 153, 102, 0, 102, 153], [102, 0, 51, 102, 153, 203, 255, 203, 203, 153, 153, 153, 102, 102, 102, 153], [102, 102, 102, 102, 153, 203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153]])
        markerbin = data == 0
        marker = label(markerbin)
        ws = watershed(data, marker, connectivity=2, watershed_line=True)
        for lab, area in zip(range(4), [34, 74, 74, 74]):
            self.assertTrue(np.sum(ws == lab) == area)

    def test_watershed_input_not_modified(self):
        """Test to ensure input markers are not modified."""
        image = np.random.default_rng().random(size=(21, 21))
        markers = np.zeros((21, 21), dtype=np.uint8)
        markers[[5, 5, 15, 15], [5, 15, 5, 15]] = [1, 2, 3, 4]
        original_markers = np.copy(markers)
        result = watershed(image, markers)
        np.testing.assert_equal(original_markers, markers)
        assert not np.all(result == markers)