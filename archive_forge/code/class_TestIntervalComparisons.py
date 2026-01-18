from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestIntervalComparisons:

    def test_interval_equal(self):
        assert Interval(0, 1) == Interval(0, 1, closed='right')
        assert Interval(0, 1) != Interval(0, 1, closed='left')
        assert Interval(0, 1) != 0

    def test_interval_comparison(self):
        msg = "'<' not supported between instances of 'pandas._libs.interval.Interval' and 'int'"
        with pytest.raises(TypeError, match=msg):
            Interval(0, 1) < 2
        assert Interval(0, 1) < Interval(1, 2)
        assert Interval(0, 1) < Interval(0, 2)
        assert Interval(0, 1) < Interval(0.5, 1.5)
        assert Interval(0, 1) <= Interval(0, 1)
        assert Interval(0, 1) > Interval(-1, 2)
        assert Interval(0, 1) >= Interval(0, 1)

    def test_equality_comparison_broadcasts_over_array(self):
        interval = Interval(0, 1)
        arr = np.array([interval, interval])
        result = interval == arr
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)