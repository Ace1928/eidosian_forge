import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
class TestPeakWidths:

    def test_empty(self):
        """
        Test if an empty array is returned if no peaks are provided.
        """
        widths = peak_widths([], [])[0]
        assert_(isinstance(widths, np.ndarray))
        assert_equal(widths.size, 0)
        widths = peak_widths([1, 2, 3], [])[0]
        assert_(isinstance(widths, np.ndarray))
        assert_equal(widths.size, 0)
        out = peak_widths([], [])
        for arr in out:
            assert_(isinstance(arr, np.ndarray))
            assert_equal(arr.size, 0)

    @pytest.mark.filterwarnings('ignore:some peaks have a width of 0')
    def test_basic(self):
        """
        Test a simple use case with easy to verify results at different relative
        heights.
        """
        x = np.array([1, 0, 1, 2, 1, 0, -1])
        prominence = 2
        for rel_height, width_true, lip_true, rip_true in [(0.0, 0.0, 3.0, 3.0), (0.25, 1.0, 2.5, 3.5), (0.5, 2.0, 2.0, 4.0), (0.75, 3.0, 1.5, 4.5), (1.0, 4.0, 1.0, 5.0), (2.0, 5.0, 1.0, 6.0), (3.0, 5.0, 1.0, 6.0)]:
            width_calc, height, lip_calc, rip_calc = peak_widths(x, [3], rel_height)
            assert_allclose(width_calc, width_true)
            assert_allclose(height, 2 - rel_height * prominence)
            assert_allclose(lip_calc, lip_true)
            assert_allclose(rip_calc, rip_true)

    def test_non_contiguous(self):
        """
        Test with non-C-contiguous input arrays.
        """
        x = np.repeat([0, 100, 50], 4)
        peaks = np.repeat([1], 3)
        result = peak_widths(x[::4], peaks[::3])
        assert_equal(result, [0.75, 75, 0.75, 1.5])

    def test_exceptions(self):
        """
        Verify that argument validation works as intended.
        """
        with raises(ValueError, match='1-D array'):
            peak_widths(np.zeros((3, 4)), np.ones(3))
        with raises(ValueError, match='1-D array'):
            peak_widths(3, [0])
        with raises(ValueError, match='1-D array'):
            peak_widths(np.arange(10), np.ones((3, 2), dtype=np.intp))
        with raises(ValueError, match='1-D array'):
            peak_widths(np.arange(10), 3)
        with raises(ValueError, match='not a valid index'):
            peak_widths(np.arange(10), [8, 11])
        with raises(ValueError, match='not a valid index'):
            peak_widths([], [1, 2])
        with raises(TypeError, match='cannot safely cast'):
            peak_widths(np.arange(10), [1.1, 2.3])
        with raises(ValueError, match='rel_height'):
            peak_widths([0, 1, 0, 1, 0], [1, 3], rel_height=-1)
        with raises(TypeError, match='None'):
            peak_widths([1, 2, 1], [1], prominence_data=(None, None, None))

    def test_warnings(self):
        """
        Verify that appropriate warnings are raised.
        """
        msg = 'some peaks have a width of 0'
        with warns(PeakPropertyWarning, match=msg):
            peak_widths([0, 1, 0], [1], rel_height=0)
        with warns(PeakPropertyWarning, match=msg):
            peak_widths([0, 1, 1, 1, 0], [2], prominence_data=(np.array([0.0], np.float64), np.array([2], np.intp), np.array([2], np.intp)))

    def test_mismatching_prominence_data(self):
        """Test with mismatching peak and / or prominence data."""
        x = [0, 1, 0]
        peak = [1]
        for i, (prominences, left_bases, right_bases) in enumerate([((1.0,), (-1,), (2,)), ((1.0,), (0,), (3,)), ((1.0,), (2,), (0,)), ((1.0, 1.0), (0, 0), (2, 2)), ((1.0, 1.0), (0,), (2,)), ((1.0,), (0, 0), (2,)), ((1.0,), (0,), (2, 2))]):
            prominence_data = (np.array(prominences, dtype=np.float64), np.array(left_bases, dtype=np.intp), np.array(right_bases, dtype=np.intp))
            if i < 3:
                match = 'prominence data is invalid for peak'
            else:
                match = 'arrays in `prominence_data` must have the same shape'
            with raises(ValueError, match=match):
                peak_widths(x, peak, prominence_data=prominence_data)

    @pytest.mark.filterwarnings('ignore:some peaks have a width of 0')
    def test_intersection_rules(self):
        """Test if x == eval_height counts as an intersection."""
        x = [0, 1, 2, 1, 3, 3, 3, 1, 2, 1, 0]
        assert_allclose(peak_widths(x, peaks=[5], rel_height=0), [(0.0,), (3.0,), (5.0,), (5.0,)])
        assert_allclose(peak_widths(x, peaks=[5], rel_height=2 / 3), [(4.0,), (1.0,), (3.0,), (7.0,)])