import pyomo.common.unittest as unittest
import pytest
import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import (
class TestIntervalToSeries(unittest.TestCase):

    def test_no_time_points(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.7, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        series_data = interval_to_series(interval_data)
        pred_time_points = [0.2, 0.5, 1.0]
        pred_data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        self.assertEqual(series_data, TimeSeriesData(pred_data, pred_time_points))

    def test_no_time_points_left_endpoints(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.7, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        series_data = interval_to_series(interval_data, use_left_endpoints=True)
        pred_time_points = [0.0, 0.2, 0.7]
        pred_data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        self.assertEqual(series_data, TimeSeriesData(pred_data, pred_time_points))

    def test_time_points_provided_no_boundary(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        time_points = [0.05 + i * 0.1 for i in range(10)]
        series_data = interval_to_series(interval_data, time_points=time_points)
        pred_data = {m.var[:, 'A']: [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], m.var[:, 'B']: [4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]}
        self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))

    def test_time_points_provided_some_on_boundary(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        time_points = [0.1 * i for i in range(11)]
        series_data = interval_to_series(interval_data, time_points=time_points)
        pred_data = {m.var[:, 'A']: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], m.var[:, 'B']: [4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]}
        self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))

    def test_time_points_provided_some_on_boundary_use_right(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        time_points = [0.1 * i for i in range(11)]
        series_data = interval_to_series(interval_data, time_points=time_points, prefer_left=False)
        pred_data = {m.var[:, 'A']: [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], m.var[:, 'B']: [4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]}
        self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))

    def test_with_roundoff_error(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        random.seed(12710)
        time_points = [i * 0.1 + random.uniform(-1e-08, 1e-08) for i in range(11)]
        series_data = interval_to_series(interval_data, time_points=time_points, tolerance=1e-07)
        pred_data = {m.var[:, 'A']: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], m.var[:, 'B']: [4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]}
        self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))