import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
class TestRidgeLines:

    def test_empty(self):
        test_matr = np.zeros([20, 100])
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        assert_(len(lines) == 0)

    def test_minimal(self):
        test_matr = np.zeros([20, 100])
        test_matr[0, 10] = 1
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        assert_(len(lines) == 1)
        test_matr = np.zeros([20, 100])
        test_matr[0:2, 10] = 1
        lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
        assert_(len(lines) == 1)

    def test_single_pass(self):
        distances = [0, 1, 2, 5]
        gaps = [0, 1, 2, 0, 1]
        test_matr = np.zeros([20, 50]) + 1e-12
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_distances = np.full(20, max(distances))
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
        assert_array_equal(identified_lines, [line])

    def test_single_bigdist(self):
        distances = [0, 1, 2, 5]
        gaps = [0, 1, 2, 4]
        test_matr = np.zeros([20, 50])
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 3
        max_distances = np.full(20, max_dist)
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
        assert_(len(identified_lines) == 2)
        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)
            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)

    def test_single_biggap(self):
        distances = [0, 1, 2, 5]
        max_gap = 3
        gaps = [0, 4, 2, 1]
        test_matr = np.zeros([20, 50])
        length = 12
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 6
        max_distances = np.full(20, max_dist)
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
        assert_(len(identified_lines) == 2)
        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)
            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)

    def test_single_biggaps(self):
        distances = [0]
        max_gap = 1
        gaps = [3, 6]
        test_matr = np.zeros([50, 50])
        length = 30
        line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
        test_matr[line[0], line[1]] = 1
        max_dist = 1
        max_distances = np.full(50, max_dist)
        identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
        assert_(len(identified_lines) == 3)
        for iline in identified_lines:
            adists = np.diff(iline[1])
            np.testing.assert_array_less(np.abs(adists), max_dist)
            agaps = np.diff(iline[0])
            np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)