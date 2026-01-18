import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
class TestMedialAxis:

    def test_00_00_zeros(self):
        """Test skeletonize on an array of all zeros"""
        result = medial_axis(np.zeros((10, 10), bool))
        assert np.all(result == False)

    def test_00_01_zeros_masked(self):
        """Test skeletonize on an array that is completely masked"""
        result = medial_axis(np.zeros((10, 10), bool), np.zeros((10, 10), bool))
        assert np.all(result == False)

    @pytest.mark.parametrize('dtype', [bool, float, int])
    def test_vertical_line(self, dtype):
        """Test a thick vertical line, issue #3861"""
        img = np.zeros((9, 9), dtype=dtype)
        img[:, 2] = 1
        img[:, 3] = 2
        img[:, 4] = 3
        expected = np.full(img.shape, False)
        expected[:, 3] = True
        result = medial_axis(img)
        assert_array_equal(result, expected)

    def test_01_01_rectangle(self):
        """Test skeletonize on a rectangle"""
        image = np.zeros((9, 15), bool)
        image[1:-1, 1:-1] = True
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        result = medial_axis(image)
        assert np.all(result == expected)
        result, distance = medial_axis(image, return_distance=True)
        assert distance.max() == 4

    def test_01_02_hole(self):
        """Test skeletonize on a rectangle with a hole in the middle"""
        image = np.zeros((9, 15), bool)
        image[1:-1, 1:-1] = True
        image[4, 4:-4] = False
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        result = medial_axis(image)
        assert np.all(result == expected)

    def test_narrow_image(self):
        """Test skeletonize on a 1-pixel thin strip"""
        image = np.zeros((1, 5), bool)
        image[:, 1:-1] = True
        result = medial_axis(image)
        assert np.all(result == image)