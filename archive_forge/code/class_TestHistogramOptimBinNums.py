import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
class TestHistogramOptimBinNums:
    """
    Provide test coverage when using provided estimators for optimal number of
    bins
    """

    def test_empty(self):
        estimator_list = ['fd', 'scott', 'rice', 'sturges', 'doane', 'sqrt', 'auto', 'stone']
        for estimator in estimator_list:
            a, b = histogram([], bins=estimator)
            assert_array_equal(a, np.array([0]))
            assert_array_equal(b, np.array([0, 1]))

    def test_simple(self):
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). All test values have been precomputed and the values
        shouldn't change
        """
        basic_test = {50: {'fd': 4, 'scott': 4, 'rice': 8, 'sturges': 7, 'doane': 8, 'sqrt': 8, 'auto': 7, 'stone': 2}, 500: {'fd': 8, 'scott': 8, 'rice': 16, 'sturges': 10, 'doane': 12, 'sqrt': 23, 'auto': 10, 'stone': 9}, 5000: {'fd': 17, 'scott': 17, 'rice': 35, 'sturges': 14, 'doane': 17, 'sqrt': 71, 'auto': 17, 'stone': 20}}
        for testlen, expectedResults in basic_test.items():
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x = np.concatenate((x1, x2))
            for estimator, numbins in expectedResults.items():
                a, b = np.histogram(x, estimator)
                assert_equal(len(a), numbins, err_msg='For the {0} estimator with datasize of {1}'.format(estimator, testlen))

    def test_small(self):
        """
        Smaller datasets have the potential to cause issues with the data
        adaptive methods, especially the FD method. All bin numbers have been
        precalculated.
        """
        small_dat = {1: {'fd': 1, 'scott': 1, 'rice': 1, 'sturges': 1, 'doane': 1, 'sqrt': 1, 'stone': 1}, 2: {'fd': 2, 'scott': 1, 'rice': 3, 'sturges': 2, 'doane': 1, 'sqrt': 2, 'stone': 1}, 3: {'fd': 2, 'scott': 2, 'rice': 3, 'sturges': 3, 'doane': 3, 'sqrt': 2, 'stone': 1}}
        for testlen, expectedResults in small_dat.items():
            testdat = np.arange(testlen)
            for estimator, expbins in expectedResults.items():
                a, b = np.histogram(testdat, estimator)
                assert_equal(len(a), expbins, err_msg='For the {0} estimator with datasize of {1}'.format(estimator, testlen))

    def test_incorrect_methods(self):
        """
        Check a Value Error is thrown when an unknown string is passed in
        """
        check_list = ['mad', 'freeman', 'histograms', 'IQR']
        for estimator in check_list:
            assert_raises(ValueError, histogram, [1, 2, 3], estimator)

    def test_novariance(self):
        """
        Check that methods handle no variance in data
        Primarily for Scott and FD as the SD and IQR are both 0 in this case
        """
        novar_dataset = np.ones(100)
        novar_resultdict = {'fd': 1, 'scott': 1, 'rice': 1, 'sturges': 1, 'doane': 1, 'sqrt': 1, 'auto': 1, 'stone': 1}
        for estimator, numbins in novar_resultdict.items():
            a, b = np.histogram(novar_dataset, estimator)
            assert_equal(len(a), numbins, err_msg='{0} estimator, No Variance test'.format(estimator))

    def test_limited_variance(self):
        """
        Check when IQR is 0, but variance exists, we return the sturges value
        and not the fd value.
        """
        lim_var_data = np.ones(1000)
        lim_var_data[:3] = 0
        lim_var_data[-4:] = 100
        edges_auto = histogram_bin_edges(lim_var_data, 'auto')
        assert_equal(edges_auto, np.linspace(0, 100, 12))
        edges_fd = histogram_bin_edges(lim_var_data, 'fd')
        assert_equal(edges_fd, np.array([0, 100]))
        edges_sturges = histogram_bin_edges(lim_var_data, 'sturges')
        assert_equal(edges_sturges, np.linspace(0, 100, 12))

    def test_outlier(self):
        """
        Check the FD, Scott and Doane with outliers.

        The FD estimates a smaller binwidth since it's less affected by
        outliers. Since the range is so (artificially) large, this means more
        bins, most of which will be empty, but the data of interest usually is
        unaffected. The Scott estimator is more affected and returns fewer bins,
        despite most of the variance being in one area of the data. The Doane
        estimator lies somewhere between the other two.
        """
        xcenter = np.linspace(-10, 10, 50)
        outlier_dataset = np.hstack((np.linspace(-110, -100, 5), xcenter))
        outlier_resultdict = {'fd': 21, 'scott': 5, 'doane': 11, 'stone': 6}
        for estimator, numbins in outlier_resultdict.items():
            a, b = np.histogram(outlier_dataset, estimator)
            assert_equal(len(a), numbins)

    def test_scott_vs_stone(self):
        """Verify that Scott's rule and Stone's rule converges for normally distributed data"""

        def nbins_ratio(seed, size):
            rng = np.random.RandomState(seed)
            x = rng.normal(loc=0, scale=2, size=size)
            a, b = (len(np.histogram(x, 'stone')[0]), len(np.histogram(x, 'scott')[0]))
            return a / (a + b)
        ll = [[nbins_ratio(seed, size) for size in np.geomspace(start=10, stop=100, num=4).round().astype(int)] for seed in range(10)]
        avg = abs(np.mean(ll, axis=0) - 0.5)
        assert_almost_equal(avg, [0.15, 0.09, 0.08, 0.03], decimal=2)

    def test_simple_range(self):
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). Adding in a 3rd mixture that will then be
        completely ignored. All test values have been precomputed and
        the shouldn't change.
        """
        basic_test = {50: {'fd': 8, 'scott': 8, 'rice': 15, 'sturges': 14, 'auto': 14, 'stone': 8}, 500: {'fd': 15, 'scott': 16, 'rice': 32, 'sturges': 20, 'auto': 20, 'stone': 80}, 5000: {'fd': 33, 'scott': 33, 'rice': 69, 'sturges': 27, 'auto': 33, 'stone': 80}}
        for testlen, expectedResults in basic_test.items():
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x3 = np.linspace(-100, -50, testlen)
            x = np.hstack((x1, x2, x3))
            for estimator, numbins in expectedResults.items():
                a, b = np.histogram(x, estimator, range=(-20, 20))
                msg = 'For the {0} estimator'.format(estimator)
                msg += ' with datasize of {0}'.format(testlen)
                assert_equal(len(a), numbins, err_msg=msg)

    @pytest.mark.parametrize('bins', ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges'])
    def test_signed_integer_data(self, bins):
        a = np.array([-2, 0, 127], dtype=np.int8)
        hist, edges = np.histogram(a, bins=bins)
        hist32, edges32 = np.histogram(a.astype(np.int32), bins=bins)
        assert_array_equal(hist, hist32)
        assert_array_equal(edges, edges32)

    def test_simple_weighted(self):
        """
        Check that weighted data raises a TypeError
        """
        estimator_list = ['fd', 'scott', 'rice', 'sturges', 'auto']
        for estimator in estimator_list:
            assert_raises(TypeError, histogram, [1, 2, 3], estimator, weights=[1, 2, 3])