import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
class TestIntervalFromTimeSeries(unittest.TestCase):

    def test_singleton(self):
        name = 'name'
        series = mpc.TimeSeriesData({name: [2.0]}, [1.0])
        interval = mpc.data.convert.series_to_interval(series)
        self.assertEqual(interval, IntervalData({name: [2.0]}, [(1.0, 1.0)]))

    def test_empty(self):
        name = 'name'
        series = mpc.TimeSeriesData({name: []}, [])
        interval = mpc.data.convert.series_to_interval(series)
        self.assertEqual(interval, mpc.IntervalData({name: []}, []))

    def test_interval_from_series(self):
        name = 'name'
        series = mpc.TimeSeriesData({name: [4.0, 5.0, 6.0]}, [1, 2, 3])
        interval = mpc.data.convert.series_to_interval(series)
        self.assertEqual(interval, mpc.IntervalData({name: [5.0, 6.0]}, [(1, 2), (2, 3)]))

    def test_use_left_endpoint(self):
        name = 'name'
        series = mpc.TimeSeriesData({name: [4.0, 5.0, 6.0]}, [1, 2, 3])
        interval = mpc.data.convert.series_to_interval(series, use_left_endpoints=True)
        self.assertEqual(interval, mpc.IntervalData({name: [4.0, 5.0]}, [(1, 2), (2, 3)]))