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
class TestProcessToDynamic(unittest.TestCase):

    def test_non_time_indexed_data(self):
        m = _make_model()
        m.scalar_var = pyo.Var(m.comp, initialize=3.0)
        data = ComponentMap([(m.scalar_var['A'], 3.1), (m.scalar_var['B'], 3.2)])
        dyn_data = _process_to_dynamic_data(data)
        self.assertTrue(isinstance(dyn_data, ScalarData))
        self.assertIn(pyo.ComponentUID(m.scalar_var['A']), dyn_data.get_data())
        self.assertIn(pyo.ComponentUID(m.scalar_var['B']), dyn_data.get_data())