import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
class TestEccentricStructuringElements:

    def setup_class(self):
        self.black_pixel = 255 * np.ones((6, 6), dtype=np.uint8)
        self.black_pixel[2, 2] = 0
        self.white_pixel = 255 - self.black_pixel
        self.footprints = [footprints.square(2), footprints.rectangle(2, 2), footprints.rectangle(2, 1), footprints.rectangle(1, 2)]

    def test_dilate_erode_symmetry(self):
        for s in self.footprints:
            c = gray.erosion(self.black_pixel, s)
            d = gray.dilation(self.white_pixel, s)
            assert np.all(c == 255 - d)

    def test_open_black_pixel(self):
        for s in self.footprints:
            gray_open = gray.opening(self.black_pixel, s)
            assert np.all(gray_open == self.black_pixel)

    def test_close_white_pixel(self):
        for s in self.footprints:
            gray_close = gray.closing(self.white_pixel, s)
            assert np.all(gray_close == self.white_pixel)

    def test_open_white_pixel(self):
        for s in self.footprints:
            assert np.all(gray.opening(self.white_pixel, s) == 0)

    def test_close_black_pixel(self):
        for s in self.footprints:
            assert np.all(gray.closing(self.black_pixel, s) == 255)

    def test_white_tophat_white_pixel(self):
        for s in self.footprints:
            tophat = gray.white_tophat(self.white_pixel, s)
            assert np.all(tophat == self.white_pixel)

    def test_black_tophat_black_pixel(self):
        for s in self.footprints:
            tophat = gray.black_tophat(self.black_pixel, s)
            assert np.all(tophat == self.white_pixel)

    def test_white_tophat_black_pixel(self):
        for s in self.footprints:
            tophat = gray.white_tophat(self.black_pixel, s)
            assert np.all(tophat == 0)

    def test_black_tophat_white_pixel(self):
        for s in self.footprints:
            tophat = gray.black_tophat(self.white_pixel, s)
            assert np.all(tophat == 0)