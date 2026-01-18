import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
class TestMquantiles:

    def test_mquantiles_limit_keyword(self):
        data = np.array([[6.0, 7.0, 1.0], [47.0, 15.0, 2.0], [49.0, 36.0, 3.0], [15.0, 39.0, 4.0], [42.0, 40.0, -999.0], [41.0, 41.0, -999.0], [7.0, -999.0, -999.0], [39.0, -999.0, -999.0], [43.0, -999.0, -999.0], [40.0, -999.0, -999.0], [36.0, -999.0, -999.0]])
        desired = [[19.2, 14.6, 1.45], [40.0, 37.5, 2.5], [42.8, 40.05, 3.55]]
        quants = mstats.mquantiles(data, axis=0, limit=(0, 50))
        assert_almost_equal(quants, desired)