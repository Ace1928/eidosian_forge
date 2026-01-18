import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
class TestPeakLocalMax:

    def test_trivial_case(self):
        trivial = np.zeros((25, 25))
        peak_indices = peak.peak_local_max(trivial, min_distance=1)
        assert type(peak_indices) is np.ndarray
        assert peak_indices.size == 0

    def test_noisy_peaks(self):
        peak_locations = [(7, 7), (7, 13), (13, 7), (13, 13)]
        image = 0.8 * np.random.rand(20, 20)
        for r, c in peak_locations:
            image[r, c] = 1
        peaks_detected = peak.peak_local_max(image, min_distance=5)
        assert len(peaks_detected) == len(peak_locations)
        for loc in peaks_detected:
            assert tuple(loc) in peak_locations

    def test_relative_threshold(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1, 1] = 10
        image[3, 3] = 20
        peaks = peak.peak_local_max(image, min_distance=1, threshold_rel=0.5)
        assert len(peaks) == 1
        assert_array_almost_equal(peaks, [(3, 3)])

    def test_absolute_threshold(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1, 1] = 10
        image[3, 3] = 20
        peaks = peak.peak_local_max(image, min_distance=1, threshold_abs=10)
        assert len(peaks) == 1
        assert_array_almost_equal(peaks, [(3, 3)])

    def test_constant_image(self):
        image = np.full((20, 20), 128, dtype=np.uint8)
        peaks = peak.peak_local_max(image, min_distance=1)
        assert len(peaks) == 0

    def test_flat_peak(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1:3, 1:3] = 10
        peaks = peak.peak_local_max(image, min_distance=1)
        assert len(peaks) == 4

    def test_sorted_peaks(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1, 1] = 20
        image[3, 3] = 10
        peaks = peak.peak_local_max(image, min_distance=1)
        assert peaks.tolist() == [[1, 1], [3, 3]]
        image = np.zeros((3, 10))
        image[1, (1, 3, 5, 7)] = (1, 2, 3, 4)
        peaks = peak.peak_local_max(image, min_distance=1)
        assert peaks.tolist() == [[1, 7], [1, 5], [1, 3], [1, 1]]

    def test_num_peaks(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1, 1] = 10
        image[1, 3] = 11
        image[1, 5] = 12
        image[3, 5] = 8
        image[5, 3] = 7
        assert len(peak.peak_local_max(image, min_distance=1, threshold_abs=0)) == 5
        peaks_limited = peak.peak_local_max(image, min_distance=1, threshold_abs=0, num_peaks=2)
        assert len(peaks_limited) == 2
        assert (1, 3) in peaks_limited
        assert (1, 5) in peaks_limited
        peaks_limited = peak.peak_local_max(image, min_distance=1, threshold_abs=0, num_peaks=4)
        assert len(peaks_limited) == 4
        assert (1, 3) in peaks_limited
        assert (1, 5) in peaks_limited
        assert (1, 1) in peaks_limited
        assert (3, 5) in peaks_limited

    def test_num_peaks_and_labels(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        labels = np.zeros((7, 7), dtype=np.uint8) + 20
        image[1, 1] = 10
        image[1, 3] = 11
        image[1, 5] = 12
        image[3, 5] = 8
        image[5, 3] = 7
        peaks_limited = peak.peak_local_max(image, min_distance=1, threshold_abs=0, labels=labels)
        assert len(peaks_limited) == 5
        peaks_limited = peak.peak_local_max(image, min_distance=1, threshold_abs=0, labels=labels, num_peaks=2)
        assert len(peaks_limited) == 2

    def test_num_peaks_tot_vs_labels_4quadrants(self):
        np.random.seed(21)
        image = np.random.uniform(size=(20, 30))
        i, j = np.mgrid[0:20, 0:30]
        labels = 1 + (i >= 10) + (j >= 15) * 2
        result = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, num_peaks=np.inf, num_peaks_per_label=2)
        assert len(result) == 8
        result = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, num_peaks=np.inf, num_peaks_per_label=1)
        assert len(result) == 4
        result = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, num_peaks=2, num_peaks_per_label=2)
        assert len(result) == 2

    def test_num_peaks3D(self):
        image = np.zeros((10, 10, 100))
        image[5, 5, ::5] = np.arange(20)
        peaks_limited = peak.peak_local_max(image, min_distance=1, num_peaks=2)
        assert len(peaks_limited) == 2

    def test_reorder_labels(self):
        image = np.random.uniform(size=(40, 60))
        i, j = np.mgrid[0:40, 0:60]
        labels = 1 + (i >= 20) + (j >= 30) * 2
        labels[labels == 4] = 5
        i, j = np.mgrid[-3:4, -3:4]
        footprint = i * i + j * j <= 9
        expected = np.zeros(image.shape, float)
        for imin, imax in ((0, 20), (20, 40)):
            for jmin, jmax in ((0, 30), (30, 60)):
                expected[imin:imax, jmin:jmax] = ndi.maximum_filter(image[imin:imax, jmin:jmax], footprint=footprint)
        expected = expected == image
        peak_idx = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, footprint=footprint, exclude_border=False)
        result = np.zeros_like(expected, dtype=bool)
        result[tuple(peak_idx.T)] = True
        assert (result == expected).all()

    def test_indices_with_labels(self):
        image = np.random.uniform(size=(40, 60))
        i, j = np.mgrid[0:40, 0:60]
        labels = 1 + (i >= 20) + (j >= 30) * 2
        i, j = np.mgrid[-3:4, -3:4]
        footprint = i * i + j * j <= 9
        expected = np.zeros(image.shape, float)
        for imin, imax in ((0, 20), (20, 40)):
            for jmin, jmax in ((0, 30), (30, 60)):
                expected[imin:imax, jmin:jmax] = ndi.maximum_filter(image[imin:imax, jmin:jmax], footprint=footprint)
        expected = np.stack(np.nonzero(expected == image), axis=-1)
        expected = expected[np.argsort(image[tuple(expected.T)])[::-1]]
        result = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, footprint=footprint, exclude_border=False)
        result = result[np.argsort(image[tuple(result.T)])[::-1]]
        assert (result == expected).all()

    def test_ndarray_exclude_border(self):
        nd_image = np.zeros((5, 5, 5))
        nd_image[[1, 0, 0], [0, 1, 0], [0, 0, 1]] = 1
        nd_image[3, 0, 0] = 1
        nd_image[2, 2, 2] = 1
        expected = np.array([[2, 2, 2]], dtype=int)
        expectedNoBorder = np.array([[0, 0, 1], [2, 2, 2], [3, 0, 0]], dtype=int)
        result = peak.peak_local_max(nd_image, min_distance=2, exclude_border=2)
        assert_array_equal(result, expected)
        assert_array_equal(peak.peak_local_max(nd_image, min_distance=2, exclude_border=2), peak.peak_local_max(nd_image, min_distance=2, exclude_border=True))
        assert_array_equal(peak.peak_local_max(nd_image, min_distance=2, exclude_border=0), peak.peak_local_max(nd_image, min_distance=2, exclude_border=False))
        result = peak.peak_local_max(nd_image, min_distance=2, exclude_border=0)
        assert_array_equal(result, expectedNoBorder)
        peak_idx = peak.peak_local_max(nd_image, exclude_border=False)
        result = np.zeros_like(nd_image, dtype=bool)
        result[tuple(peak_idx.T)] = True
        assert_array_equal(result, nd_image.astype(bool))

    def test_empty(self):
        image = np.zeros((10, 20))
        labels = np.zeros((10, 20), int)
        result = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
        assert result.shape == (0, image.ndim)

    def test_empty_non2d_indices(self):
        image = np.zeros((10, 10, 10))
        result = peak.peak_local_max(image, footprint=np.ones((3, 3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
        assert result.shape == (0, image.ndim)

    def test_one_point(self):
        image = np.zeros((10, 20))
        labels = np.zeros((10, 20), int)
        image[5, 5] = 1
        labels[5, 5] = 1
        peak_idx = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
        result = np.zeros_like(image, dtype=bool)
        result[tuple(peak_idx.T)] = True
        assert np.all(result == (labels == 1))

    def test_adjacent_and_same(self):
        image = np.zeros((10, 20))
        labels = np.zeros((10, 20), int)
        image[5, 5:6] = 1
        labels[5, 5:6] = 1
        expected = np.stack(np.where(labels == 1), axis=-1)
        result = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
        assert_array_equal(result, expected)

    def test_adjacent_and_different(self):
        image = np.zeros((10, 20))
        labels = np.zeros((10, 20), int)
        image[5, 5] = 1
        image[5, 6] = 0.5
        labels[5, 5:6] = 1
        expected = np.stack(np.where(image == 1), axis=-1)
        result = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
        assert_array_equal(result, expected)
        result = peak.peak_local_max(image, labels=labels, min_distance=1, threshold_rel=0, exclude_border=False)
        assert_array_equal(result, expected)

    def test_not_adjacent_and_different(self):
        image = np.zeros((10, 20))
        labels = np.zeros((10, 20), int)
        image[5, 5] = 1
        image[5, 8] = 0.5
        labels[image > 0] = 1
        expected = np.stack(np.where(labels == 1), axis=-1)
        result = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
        assert_array_equal(result, expected)

    def test_two_objects(self):
        image = np.zeros((10, 20))
        labels = np.zeros((10, 20), int)
        image[5, 5] = 1
        image[5, 15] = 0.5
        labels[5, 5] = 1
        labels[5, 15] = 2
        expected = np.stack(np.where(labels > 0), axis=-1)
        result = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
        assert_array_equal(result, expected)

    def test_adjacent_different_objects(self):
        image = np.zeros((10, 20))
        labels = np.zeros((10, 20), int)
        image[5, 5] = 1
        image[5, 6] = 0.5
        labels[5, 5] = 1
        labels[5, 6] = 2
        expected = np.stack(np.where(labels > 0), axis=-1)
        result = peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
        assert_array_equal(result, expected)

    def test_four_quadrants(self):
        image = np.random.uniform(size=(20, 30))
        i, j = np.mgrid[0:20, 0:30]
        labels = 1 + (i >= 10) + (j >= 15) * 2
        i, j = np.mgrid[-3:4, -3:4]
        footprint = i * i + j * j <= 9
        expected = np.zeros(image.shape, float)
        for imin, imax in ((0, 10), (10, 20)):
            for jmin, jmax in ((0, 15), (15, 30)):
                expected[imin:imax, jmin:jmax] = ndi.maximum_filter(image[imin:imax, jmin:jmax], footprint=footprint)
        expected = expected == image
        peak_idx = peak.peak_local_max(image, labels=labels, footprint=footprint, min_distance=1, threshold_rel=0, exclude_border=False)
        result = np.zeros_like(image, dtype=bool)
        result[tuple(peak_idx.T)] = True
        assert np.all(result == expected)

    def test_disk(self):
        """regression test of img-1194, footprint = [1]
        Test peak.peak_local_max when every point is a local maximum
        """
        image = np.random.uniform(size=(10, 20))
        footprint = np.array([[1]])
        peak_idx = peak.peak_local_max(image, labels=np.ones((10, 20), int), footprint=footprint, min_distance=1, threshold_rel=0, threshold_abs=-1, exclude_border=False)
        result = np.zeros_like(image, dtype=bool)
        result[tuple(peak_idx.T)] = True
        assert np.all(result)
        peak_idx = peak.peak_local_max(image, footprint=footprint, threshold_abs=-1, exclude_border=False)
        result = np.zeros_like(image, dtype=bool)
        result[tuple(peak_idx.T)] = True
        assert np.all(result)

    def test_3D(self):
        image = np.zeros((30, 30, 30))
        image[15, 15, 15] = 1
        image[5, 5, 5] = 1
        assert_array_equal(peak.peak_local_max(image, min_distance=10, threshold_rel=0), [[15, 15, 15]])
        assert_array_equal(peak.peak_local_max(image, min_distance=6, threshold_rel=0), [[15, 15, 15]])
        assert sorted(peak.peak_local_max(image, min_distance=10, threshold_rel=0, exclude_border=False).tolist()) == [[5, 5, 5], [15, 15, 15]]
        assert sorted(peak.peak_local_max(image, min_distance=5, threshold_rel=0).tolist()) == [[5, 5, 5], [15, 15, 15]]

    def test_4D(self):
        image = np.zeros((30, 30, 30, 30))
        image[15, 15, 15, 15] = 1
        image[5, 5, 5, 5] = 1
        assert_array_equal(peak.peak_local_max(image, min_distance=10, threshold_rel=0), [[15, 15, 15, 15]])
        assert_array_equal(peak.peak_local_max(image, min_distance=6, threshold_rel=0), [[15, 15, 15, 15]])
        assert sorted(peak.peak_local_max(image, min_distance=10, threshold_rel=0, exclude_border=False).tolist()) == [[5, 5, 5, 5], [15, 15, 15, 15]]
        assert sorted(peak.peak_local_max(image, min_distance=5, threshold_rel=0).tolist()) == [[5, 5, 5, 5], [15, 15, 15, 15]]

    def test_threshold_rel_default(self):
        image = np.ones((5, 5))
        image[2, 2] = 1
        assert len(peak.peak_local_max(image)) == 0
        image[2, 2] = 2
        assert_array_equal(peak.peak_local_max(image), [[2, 2]])
        image[2, 2] = 0
        with expected_warnings(['When min_distance < 1']):
            assert len(peak.peak_local_max(image, min_distance=0)) == image.size - 1

    def test_peak_at_border(self):
        image = np.full((10, 10), -2)
        image[2, 4] = -1
        image[3, 0] = -1
        peaks = peak.peak_local_max(image, min_distance=3)
        assert peaks.size == 0
        peaks = peak.peak_local_max(image, min_distance=3, exclude_border=0)
        assert len(peaks) == 2
        assert [2, 4] in peaks
        assert [3, 0] in peaks