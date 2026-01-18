import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
class TestLocalMaxima(unittest.TestCase):
    """Some tests for local_minima are included as well."""
    supported_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
    image = np.array([[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 2, 0, 0, 3, 3, 0, 0, 4, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 4, 4, 0, 3, 0, 0, 0], [0, 2, 0, 1, 0, 2, 1, 0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 2, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0]], dtype=np.uint8)
    expected_default = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=bool)
    expected_cross = np.array([[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=bool)

    def test_empty(self):
        """Test result with empty image."""
        result = extrema.local_maxima(np.array([[]]), indices=False)
        assert result.size == 0
        assert result.dtype == bool
        assert result.shape == (1, 0)
        result = extrema.local_maxima(np.array([]), indices=True)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].size == 0
        assert result[0].dtype == np.intp
        result = extrema.local_maxima(np.array([[]]), indices=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].size == 0
        assert result[0].dtype == np.intp
        assert result[1].size == 0
        assert result[1].dtype == np.intp

    def test_dtypes(self):
        """Test results with default configuration for all supported dtypes."""
        for dtype in self.supported_dtypes:
            result = extrema.local_maxima(self.image.astype(dtype))
            assert result.dtype == bool
            assert_equal(result, self.expected_default)

    def test_dtypes_old(self):
        """
        Test results with default configuration and data copied from old unit
        tests for all supported dtypes.
        """
        data = np.array([[10, 11, 13, 14, 14, 15, 14, 14, 13, 11], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13], [13, 15, 40, 40, 18, 18, 18, 60, 60, 15], [14, 16, 40, 40, 19, 19, 19, 60, 60, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [15, 16, 18, 19, 19, 20, 19, 19, 18, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [14, 16, 80, 80, 19, 19, 19, 100, 100, 16], [13, 15, 80, 80, 18, 18, 18, 100, 100, 15], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]], dtype=np.uint8)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        for dtype in self.supported_dtypes:
            image = data.astype(dtype)
            result = extrema.local_maxima(image)
            assert result.dtype == bool
            assert_equal(result, expected)

    def test_connectivity(self):
        """Test results if footprint is a scalar."""
        result_conn1 = extrema.local_maxima(self.image, connectivity=1)
        assert result_conn1.dtype == bool
        assert_equal(result_conn1, self.expected_cross)
        result_conn2 = extrema.local_maxima(self.image, connectivity=2)
        assert result_conn2.dtype == bool
        assert_equal(result_conn2, self.expected_default)
        result_conn3 = extrema.local_maxima(self.image, connectivity=3)
        assert result_conn3.dtype == bool
        assert_equal(result_conn3, self.expected_default)

    def test_footprint(self):
        """Test results if footprint is given."""
        footprint_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        result_footprint_cross = extrema.local_maxima(self.image, footprint=footprint_cross)
        assert result_footprint_cross.dtype == bool
        assert_equal(result_footprint_cross, self.expected_cross)
        for footprint in [((True,) * 3,) * 3, np.ones((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.uint8), np.ones((3, 3), dtype=bool)]:
            result_footprint_square = extrema.local_maxima(self.image, footprint=footprint)
            assert result_footprint_square.dtype == bool
            assert_equal(result_footprint_square, self.expected_default)
        footprint_x = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
        expected_footprint_x = np.array([[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=bool)
        result_footprint_x = extrema.local_maxima(self.image, footprint=footprint_x)
        assert result_footprint_x.dtype == bool
        assert_equal(result_footprint_x, expected_footprint_x)

    def test_indices(self):
        """Test output if indices of peaks are desired."""
        expected_conn1 = np.nonzero(self.expected_cross)
        result_conn1 = extrema.local_maxima(self.image, connectivity=1, indices=True)
        assert_equal(result_conn1, expected_conn1)
        expected_conn2 = np.nonzero(self.expected_default)
        result_conn2 = extrema.local_maxima(self.image, connectivity=2, indices=True)
        assert_equal(result_conn2, expected_conn2)

    def test_allow_borders(self):
        """Test maxima detection at the image border."""
        result_with_boder = extrema.local_maxima(self.image, connectivity=1, allow_borders=True)
        assert result_with_boder.dtype == bool
        assert_equal(result_with_boder, self.expected_cross)
        expected_without_border = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        result_without_border = extrema.local_maxima(self.image, connectivity=1, allow_borders=False)
        assert result_with_boder.dtype == bool
        assert_equal(result_without_border, expected_without_border)

    def test_nd(self):
        """Test one- and three-dimensional case."""
        x_1d = np.array([1, 1, 0, 1, 2, 3, 0, 2, 1, 2, 0])
        expected_1d = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        result_1d = extrema.local_maxima(x_1d)
        assert result_1d.dtype == bool
        assert_equal(result_1d, expected_1d)
        x_3d = np.zeros((8, 8, 8), dtype=np.uint8)
        expected_3d = np.zeros((8, 8, 8), dtype=bool)
        x_3d[1, 1:3, 1:3] = 100
        x_3d[2, 2, 2] = 200
        x_3d[3, 1:3, 1:3] = 100
        expected_3d[2, 2, 2] = 1
        x_3d[5:8, 1, 1] = 200
        expected_3d[5:8, 1, 1] = 1
        x_3d[0, 5:8, 5:8] = 200
        x_3d[1, 6, 6] = 100
        x_3d[2, 5:7, 5:7] = 200
        x_3d[0:3, 5:8, 5:8] += 50
        expected_3d[0, 5:8, 5:8] = 1
        expected_3d[2, 5:7, 5:7] = 1
        x_3d[6:8, 6:8, 6:8] = 200
        x_3d[7, 7, 7] = 255
        expected_3d[7, 7, 7] = 1
        result_3d = extrema.local_maxima(x_3d)
        assert result_3d.dtype == bool
        assert_equal(result_3d, expected_3d)

    def test_constant(self):
        """Test behaviour for 'flat' images."""
        const_image = np.full((7, 6), 42, dtype=np.uint8)
        expected = np.zeros((7, 6), dtype=np.uint8)
        for dtype in self.supported_dtypes:
            const_image = const_image.astype(dtype)
            result = extrema.local_maxima(const_image)
            assert result.dtype == bool
            assert_equal(result, expected)
            result = extrema.local_minima(const_image)
            assert result.dtype == bool
            assert_equal(result, expected)

    def test_extrema_float(self):
        """Specific tests for float type."""
        image = np.array([[0.1, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14, 0.14, 0.13, 0.11], [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13], [0.13, 0.15, 0.4, 0.4, 0.18, 0.18, 0.18, 0.6, 0.6, 0.15], [0.14, 0.16, 0.4, 0.4, 0.19, 0.19, 0.19, 0.6, 0.6, 0.16], [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16], [0.15, 0.182, 0.18, 0.19, 0.204, 0.2, 0.19, 0.19, 0.18, 0.16], [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16], [0.14, 0.16, 0.8, 0.8, 0.19, 0.19, 0.19, 1.0, 1.0, 0.16], [0.13, 0.15, 0.8, 0.8, 0.18, 0.18, 0.18, 1.0, 1.0, 0.15], [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13]], dtype=np.float32)
        inverted_image = 1.0 - image
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        result = extrema.local_maxima(image)
        assert result.dtype == bool
        assert_equal(result, expected_result)
        result = extrema.local_minima(inverted_image)
        assert result.dtype == bool
        assert_equal(result, expected_result)

    def test_exceptions(self):
        """Test if input validation triggers correct exceptions."""
        with raises(ValueError, match='number of dimensions'):
            extrema.local_maxima(self.image, footprint=np.ones((3, 3, 3), dtype=bool))
        with raises(ValueError, match='number of dimensions'):
            extrema.local_maxima(self.image, footprint=np.ones((3,), dtype=bool))
        with raises(ValueError, match='dimension size'):
            extrema.local_maxima(self.image, footprint=np.ones((2, 3), dtype=bool))
        with raises(ValueError, match='dimension size'):
            extrema.local_maxima(self.image, footprint=np.ones((5, 5), dtype=bool))
        with raises(TypeError, match='float16 which is not supported'):
            extrema.local_maxima(np.empty(1, dtype=np.float16))

    def test_small_array(self):
        """Test output for arrays with dimension smaller 3.

        If any dimension of an array is smaller than 3 and `allow_borders` is
        false a footprint, which has at least 3 elements in each
        dimension, can't be applied. This is an implementation detail so
        `local_maxima` should still return valid output (see gh-3261).

        If `allow_borders` is true the array is padded internally and there is
        no problem.
        """
        warning_msg = "maxima can't exist .* any dimension smaller 3 .*"
        x = np.array([0, 1])
        extrema.local_maxima(x, allow_borders=True)
        with warns(UserWarning, match=warning_msg):
            result = extrema.local_maxima(x, allow_borders=False)
        assert_equal(result, [0, 0])
        assert result.dtype == bool
        x = np.array([[1, 2], [2, 2]])
        extrema.local_maxima(x, allow_borders=True, indices=True)
        with warns(UserWarning, match=warning_msg):
            result = extrema.local_maxima(x, allow_borders=False, indices=True)
        assert_equal(result, np.zeros((2, 0), dtype=np.intp))
        assert result[0].dtype == np.intp
        assert result[1].dtype == np.intp