import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
class TestPulse(CommonTests):

    def test_bad_interval_var(self):
        with self.assertRaisesRegex(TypeError, "The 'interval_var' argument for a 'Pulse' must be an 'IntervalVar'.\nReceived: <class 'float'>"):
            thing = Pulse(interval_var=1.2, height=4)

    def test_create_pulse_with_scalar_interval_var(self):
        m = self.get_model()
        p = Pulse(interval_var=m.a, height=1)
        self.assertIsInstance(p, Pulse)
        self.assertEqual(str(p), 'Pulse(a, height=1)')

    def test_create_pulse_with_interval_var_data(self):
        m = self.get_model()
        p = Pulse(interval_var=m.c[2], height=2)
        self.assertIsInstance(p, Pulse)
        self.assertEqual(str(p), 'Pulse(c[2], height=2)')