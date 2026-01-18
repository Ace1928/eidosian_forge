import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
class TestFindPeaks:
    property_keys = {'peak_heights', 'left_thresholds', 'right_thresholds', 'prominences', 'left_bases', 'right_bases', 'widths', 'width_heights', 'left_ips', 'right_ips'}

    def test_constant(self):
        """
        Test behavior for signal without local maxima.
        """
        open_interval = (None, None)
        peaks, props = find_peaks(np.ones(10), height=open_interval, threshold=open_interval, prominence=open_interval, width=open_interval)
        assert_(peaks.size == 0)
        for key in self.property_keys:
            assert_(props[key].size == 0)

    def test_plateau_size(self):
        """
        Test plateau size condition for peaks.
        """
        plateau_sizes = np.array([1, 2, 3, 4, 8, 20, 111])
        x = np.zeros(plateau_sizes.size * 2 + 1)
        x[1::2] = plateau_sizes
        repeats = np.ones(x.size, dtype=int)
        repeats[1::2] = x[1::2]
        x = np.repeat(x, repeats)
        peaks, props = find_peaks(x, plateau_size=(None, None))
        assert_equal(peaks, [1, 3, 7, 11, 18, 33, 100])
        assert_equal(props['plateau_sizes'], plateau_sizes)
        assert_equal(props['left_edges'], peaks - (plateau_sizes - 1) // 2)
        assert_equal(props['right_edges'], peaks + plateau_sizes // 2)
        assert_equal(find_peaks(x, plateau_size=4)[0], [11, 18, 33, 100])
        assert_equal(find_peaks(x, plateau_size=(None, 3.5))[0], [1, 3, 7])
        assert_equal(find_peaks(x, plateau_size=(5, 50))[0], [18, 33])

    def test_height_condition(self):
        """
        Test height condition for peaks.
        """
        x = (0.0, 1 / 3, 0.0, 2.5, 0, 4.0, 0)
        peaks, props = find_peaks(x, height=(None, None))
        assert_equal(peaks, np.array([1, 3, 5]))
        assert_equal(props['peak_heights'], np.array([1 / 3, 2.5, 4.0]))
        assert_equal(find_peaks(x, height=0.5)[0], np.array([3, 5]))
        assert_equal(find_peaks(x, height=(None, 3))[0], np.array([1, 3]))
        assert_equal(find_peaks(x, height=(2, 3))[0], np.array([3]))

    def test_threshold_condition(self):
        """
        Test threshold condition for peaks.
        """
        x = (0, 2, 1, 4, -1)
        peaks, props = find_peaks(x, threshold=(None, None))
        assert_equal(peaks, np.array([1, 3]))
        assert_equal(props['left_thresholds'], np.array([2, 3]))
        assert_equal(props['right_thresholds'], np.array([1, 5]))
        assert_equal(find_peaks(x, threshold=2)[0], np.array([3]))
        assert_equal(find_peaks(x, threshold=3.5)[0], np.array([]))
        assert_equal(find_peaks(x, threshold=(None, 5))[0], np.array([1, 3]))
        assert_equal(find_peaks(x, threshold=(None, 4))[0], np.array([1]))
        assert_equal(find_peaks(x, threshold=(2, 4))[0], np.array([]))

    def test_distance_condition(self):
        """
        Test distance condition for peaks.
        """
        peaks_all = np.arange(1, 21, 3)
        x = np.zeros(21)
        x[peaks_all] += np.linspace(1, 2, peaks_all.size)
        assert_equal(find_peaks(x, distance=3)[0], peaks_all)
        peaks_subset = find_peaks(x, distance=3.0001)[0]
        assert_(np.setdiff1d(peaks_subset, peaks_all, assume_unique=True).size == 0)
        assert_equal(np.diff(peaks_subset), 6)
        x = [-2, 1, -1, 0, -3]
        peaks_subset = find_peaks(x, distance=10)[0]
        assert_(peaks_subset.size == 1 and peaks_subset[0] == 1)

    def test_prominence_condition(self):
        """
        Test prominence condition for peaks.
        """
        x = np.linspace(0, 10, 100)
        peaks_true = np.arange(1, 99, 2)
        offset = np.linspace(1, 10, peaks_true.size)
        x[peaks_true] += offset
        prominences = x[peaks_true] - x[peaks_true + 1]
        interval = (3, 9)
        keep = np.nonzero((interval[0] <= prominences) & (prominences <= interval[1]))
        peaks_calc, properties = find_peaks(x, prominence=interval)
        assert_equal(peaks_calc, peaks_true[keep])
        assert_equal(properties['prominences'], prominences[keep])
        assert_equal(properties['left_bases'], 0)
        assert_equal(properties['right_bases'], peaks_true[keep] + 1)

    def test_width_condition(self):
        """
        Test width condition for peaks.
        """
        x = np.array([1, 0, 1, 2, 1, 0, -1, 4, 0])
        peaks, props = find_peaks(x, width=(None, 2), rel_height=0.75)
        assert_equal(peaks.size, 1)
        assert_equal(peaks, 7)
        assert_allclose(props['widths'], 1.35)
        assert_allclose(props['width_heights'], 1.0)
        assert_allclose(props['left_ips'], 6.4)
        assert_allclose(props['right_ips'], 7.75)

    def test_properties(self):
        """
        Test returned properties.
        """
        open_interval = (None, None)
        x = [0, 1, 0, 2, 1.5, 0, 3, 0, 5, 9]
        peaks, props = find_peaks(x, height=open_interval, threshold=open_interval, prominence=open_interval, width=open_interval)
        assert_(len(props) == len(self.property_keys))
        for key in self.property_keys:
            assert_(peaks.size == props[key].size)

    def test_raises(self):
        """
        Test exceptions raised by function.
        """
        with raises(ValueError, match='1-D array'):
            find_peaks(np.array(1))
        with raises(ValueError, match='1-D array'):
            find_peaks(np.ones((2, 2)))
        with raises(ValueError, match='distance'):
            find_peaks(np.arange(10), distance=-1)

    @pytest.mark.filterwarnings('ignore:some peaks have a prominence of 0', 'ignore:some peaks have a width of 0')
    def test_wlen_smaller_plateau(self):
        """
        Test behavior of prominence and width calculation if the given window
        length is smaller than a peak's plateau size.

        Regression test for gh-9110.
        """
        peaks, props = find_peaks([0, 1, 1, 1, 0], prominence=(None, None), width=(None, None), wlen=2)
        assert_equal(peaks, 2)
        assert_equal(props['prominences'], 0)
        assert_equal(props['widths'], 0)
        assert_equal(props['width_heights'], 1)
        for key in ('left_bases', 'right_bases', 'left_ips', 'right_ips'):
            assert_equal(props[key], peaks)

    @pytest.mark.parametrize('kwargs', [{}, {'distance': 3.0}, {'prominence': (None, None)}, {'width': (None, 2)}])
    def test_readonly_array(self, kwargs):
        """
        Test readonly arrays are accepted.
        """
        x = np.linspace(0, 10, 15)
        x_readonly = x.copy()
        x_readonly.flags.writeable = False
        peaks, _ = find_peaks(x)
        peaks_readonly, _ = find_peaks(x_readonly, **kwargs)
        assert_allclose(peaks, peaks_readonly)