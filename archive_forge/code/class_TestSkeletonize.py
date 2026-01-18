import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
class TestSkeletonize:

    def test_skeletonize_no_foreground(self):
        im = np.zeros((5, 5))
        result = skeletonize(im)
        assert_array_equal(result, np.zeros((5, 5)))

    def test_skeletonize_wrong_dim1(self):
        im = np.zeros(5, dtype=bool)
        with pytest.raises(ValueError):
            skeletonize(im)

    def test_skeletonize_wrong_dim2(self):
        im = np.zeros((5, 5, 5), dtype=bool)
        with pytest.raises(ValueError):
            skeletonize(im, method='zhang')

    def test_skeletonize_wrong_method(self):
        im = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError):
            skeletonize(im, method='foo')

    def test_skeletonize_all_foreground(self):
        im = np.ones((3, 4), dtype=bool)
        skeletonize(im)

    def test_skeletonize_single_point(self):
        im = np.zeros((5, 5), dtype=bool)
        im[3, 3] = 1
        result = skeletonize(im)
        assert_array_equal(result, im)

    def test_skeletonize_already_thinned(self):
        im = np.zeros((5, 5), dtype=bool)
        im[3, 1:-1] = 1
        im[2, -1] = 1
        im[4, 0] = 1
        result = skeletonize(im)
        assert_array_equal(result, im)

    def test_skeletonize_output(self):
        im = imread(fetch('data/bw_text.png'), as_gray=True)
        im = im == 0
        result = skeletonize(im)
        expected = np.load(fetch('data/bw_text_skeleton.npy'))
        assert_array_equal(result, expected)

    @pytest.mark.parametrize('dtype', [bool, float, int])
    def test_skeletonize_num_neighbors(self, dtype):
        image = np.zeros((300, 300), dtype=dtype)
        image[10:-10, 10:100] = 2
        image[-100:-10, 10:-10] = 2
        image[10:-10, -100:-10] = 2
        rs, cs = draw.line(250, 150, 10, 280)
        for i in range(10):
            image[rs + i, cs] = 1
        rs, cs = draw.line(10, 150, 250, 280)
        for i in range(20):
            image[rs + i, cs] = 3
        ir, ic = np.indices(image.shape)
        circle1 = (ic - 135) ** 2 + (ir - 150) ** 2 < 30 ** 2
        circle2 = (ic - 135) ** 2 + (ir - 150) ** 2 < 20 ** 2
        image[circle1] = 1
        image[circle2] = 0
        result = skeletonize(image)
        mask = np.array([[1, 1], [1, 1]], np.uint8)
        blocks = correlate(result, mask, mode='constant')
        assert not np.any(blocks == 4)

    def test_lut_fix(self):
        im = np.zeros((6, 6), dtype=bool)
        im[1, 2] = 1
        im[2, 2] = 1
        im[2, 3] = 1
        im[3, 3] = 1
        im[3, 4] = 1
        im[4, 4] = 1
        im[4, 5] = 1
        result = skeletonize(im)
        expected = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], dtype=bool)
        assert np.all(result == expected)

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_skeletonize_copies_input(self, ndim):
        """Skeletonize mustn't modify the original input image."""
        image = np.ones((3,) * ndim, dtype=bool)
        image = np.pad(image, 1)
        original = image.copy()
        skeletonize(image)
        np.testing.assert_array_equal(image, original)