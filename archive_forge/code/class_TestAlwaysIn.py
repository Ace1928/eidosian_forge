import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
class TestAlwaysIn(CommonTests):

    def test_always_in(self):
        m = self.get_model()
        f = Pulse(interval_var=m.a, height=3) + Step(m.b.start_time, height=2) - Step(m.a.end_time, height=-1)
        m.cons = LogicalConstraint(expr=f.within((0, 3), (0, 10)))
        self.assertIsInstance(m.cons.expr, AlwaysIn)
        self.assertEqual(m.cons.expr.nargs(), 5)
        self.assertEqual(len(m.cons.expr.args), 5)
        self.assertIs(m.cons.expr.args[0], f)
        self.assertEqual(m.cons.expr.args[1], 0)
        self.assertEqual(m.cons.expr.args[2], 3)
        self.assertEqual(m.cons.expr.args[3], 0)
        self.assertEqual(m.cons.expr.args[4], 10)
        self.assertEqual(str(m.cons.expr), '(Pulse(a, height=3) + Step(b.start_time, height=2) - Step(a.end_time, height=-1)).within(bounds=(0, 3), times=(0, 10))')