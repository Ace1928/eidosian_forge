import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
class TestIntervalData(unittest.TestCase):

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0.1 * i for i in range(11)])
        m.comp = pyo.Set(initialize=['A', 'B'])
        m.var = pyo.Var(m.time, m.comp, initialize=1.0)
        return m

    def test_construct(self):
        m = self._make_model()
        intervals = [(0.0, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0], m.var[:, 'B']: [3.0, 4.0]}
        interval_data = IntervalData(data, intervals)
        self.assertEqual(interval_data.get_data(), {pyo.ComponentUID(key): val for key, val in data.items()})
        self.assertEqual(intervals, interval_data.get_intervals())

    def test_eq(self):
        m = self._make_model()
        intervals = [(0.0, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0], m.var[:, 'B']: [3.0, 4.0]}
        interval_data_1 = IntervalData(data, intervals)
        data = {m.var[:, 'A']: [1.0, 2.0], m.var[:, 'B']: [3.0, 4.0]}
        interval_data_2 = IntervalData(data, intervals)
        self.assertEqual(interval_data_1, interval_data_2)
        data = {m.var[:, 'A']: [1.0, 3.0], m.var[:, 'B']: [3.0, 4.0]}
        interval_data_3 = IntervalData(data, intervals)
        self.assertNotEqual(interval_data_1, interval_data_3)

    def test_get_data_at_indices_multiple(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        data = interval_data.get_data_at_interval_indices([0, 2])
        pred_data = IntervalData({m.var[:, 'A']: [1.0, 3.0], m.var[:, 'B']: [4.0, 6.0]}, [(0.0, 0.2), (0.5, 1.0)])
        self.assertEqual(pred_data, data)

    def test_get_data_at_indices_singleton(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        data = interval_data.get_data_at_interval_indices(1)
        pred_data = ScalarData({m.var[:, 'A']: 2.0, m.var[:, 'B']: 5.0})
        self.assertEqual(data, pred_data)

    def test_get_data_at_time_scalar(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        data = interval_data.get_data_at_time(0.1)
        pred_data = ScalarData({m.var[:, 'A']: 1.0, m.var[:, 'B']: 4.0})
        self.assertEqual(data, pred_data)
        data = interval_data.get_data_at_time(1.1)
        pred_data = ScalarData({m.var[:, 'A']: 3.0, m.var[:, 'B']: 6.0})
        self.assertEqual(data, pred_data)
        msg = 'Time point.*not found'
        with self.assertRaisesRegex(RuntimeError, msg):
            data = interval_data.get_data_at_time(1.1, tolerance=0.001)
        data = interval_data.get_data_at_time(0.5)
        pred_data = ScalarData({m.var[:, 'A']: 2.0, m.var[:, 'B']: 5.0})
        self.assertEqual(data, pred_data)
        data = interval_data.get_data_at_time(0.5, prefer_left=False)
        pred_data = ScalarData({m.var[:, 'A']: 3.0, m.var[:, 'B']: 6.0})
        self.assertEqual(data, pred_data)

    def test_to_serializable(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        json_data = interval_data.to_serializable()
        self.assertEqual(json_data, ({'var[*,A]': [1.0, 2.0, 3.0], 'var[*,B]': [4.0, 5.0, 6.0]}, [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]))

    def test_concatenate(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data_1 = IntervalData(data, intervals)
        intervals = [(1.0, 1.5), (2.0, 3.0)]
        data = {m.var[:, 'A']: [7.0, 8.0], m.var[:, 'B']: [9.0, 10.0]}
        interval_data_2 = IntervalData(data, intervals)
        interval_data_1.concatenate(interval_data_2)
        new_intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (2.0, 3.0)]
        new_values = {m.var[:, 'A']: [1.0, 2.0, 3.0, 7.0, 8.0], m.var[:, 'B']: [4.0, 5.0, 6.0, 9.0, 10.0]}
        new_data = IntervalData(new_values, new_intervals)
        self.assertEqual(interval_data_1, new_data)

    def test_shift_time_points(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        interval_data.shift_time_points(1.0)
        intervals = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        new_interval_data = IntervalData(data, intervals)
        self.assertEqual(interval_data, new_interval_data)

    def test_extract_variables(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals, time_set=m.time)
        new_data = interval_data.extract_variables([m.var[:, 'B']])
        value_dict = {m.var[:, 'B']: [4.0, 5.0, 6.0]}
        pred_data = IntervalData(value_dict, intervals)
        self.assertEqual(new_data, pred_data)

    def test_extract_variables_exception(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals, time_set=m.time)
        msg = 'only accepts a list or tuple'
        with self.assertRaisesRegex(TypeError, msg):
            new_data = interval_data.extract_variables(m.var[:, 'B'])