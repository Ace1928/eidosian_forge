import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
class TestSubtractStepFunctions(CommonTests):

    def test_subtract_two_steps(self):
        m = self.get_model()
        s = Step(m.a.start_time, height=2) - Step(m.b.start_time, height=5)
        self.assertIsInstance(s, CumulativeFunction)
        self.assertEqual(len(s.args), 2)
        self.assertEqual(s.nargs(), 2)
        self.assertIsInstance(s.args[0], StepAtStart)
        self.assertIsInstance(s.args[1], NegatedStepFunction)
        self.assertEqual(len(s.args[1].args), 1)
        self.assertEqual(s.args[1].nargs(), 1)
        self.assertIsInstance(s.args[1].args[0], StepAtStart)

    def test_subtract_step_and_pulse(self):
        m = self.get_model()
        s1 = Step(m.a.end_time, height=2)
        s2 = Step(m.b.start_time, height=5)
        p = Pulse(interval_var=m.a, height=3)
        expr = s1 - s2 - p
        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(len(expr.args), 3)
        self.assertEqual(expr.nargs(), 3)
        self.assertIs(expr.args[0], s1)
        self.assertIsInstance(expr.args[1], NegatedStepFunction)
        self.assertIs(expr.args[1].args[0], s2)
        self.assertIsInstance(expr.args[2], NegatedStepFunction)
        self.assertIs(expr.args[2].args[0], p)

    def test_subtract_pulse_from_two_steps(self):
        m = self.get_model()
        s1 = Step(m.a.end_time, height=2)
        s2 = Step(m.b.start_time, height=5)
        p = Pulse(interval_var=m.a, height=3)
        expr = s1 + s2 - p
        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(len(expr.args), 3)
        self.assertEqual(expr.nargs(), 3)
        self.assertIs(expr.args[0], s1)
        self.assertIs(expr.args[1], s2)
        self.assertIsInstance(expr.args[2], NegatedStepFunction)
        self.assertIs(expr.args[2].args[0], p)

    def test_args_clone_correctly(self):
        m = self.get_model()
        m.p1 = Pulse(interval_var=m.a, height=3)
        m.p2 = Pulse(interval_var=m.b, height=4)
        m.s = Step(m.a.start_time, height=-1)
        expr1 = m.p1 - m.p2
        self.assertIsInstance(expr1, CumulativeFunction)
        self.assertEqual(expr1.nargs(), 2)
        self.assertIs(expr1.args[0], m.p1)
        self.assertIsInstance(expr1.args[1], NegatedStepFunction)
        self.assertIs(expr1.args[1].args[0], m.p2)
        expr2 = m.p1 - m.s
        self.assertIsInstance(expr2, CumulativeFunction)
        self.assertEqual(expr2.nargs(), 2)
        self.assertIs(expr2.args[0], m.p1)
        self.assertIsInstance(expr2.args[1], NegatedStepFunction)
        self.assertIs(expr2.args[1].args[0], m.s)

    def test_args_clone_correctly_in_place(self):
        m = self.get_model()
        m.p1 = Pulse(interval_var=m.a, height=3)
        m.p2 = Pulse(interval_var=m.b, height=4)
        m.s = Step(m.a.start_time, height=-1)
        expr1 = m.p1 - m.p2
        expr = expr1 + m.p1
        expr1 -= m.s
        self.assertIsInstance(expr1, CumulativeFunction)
        self.assertEqual(expr1.nargs(), 3)
        self.assertIs(expr1.args[0], m.p1)
        self.assertIsInstance(expr1.args[1], NegatedStepFunction)
        self.assertIs(expr1.args[1].args[0], m.p2)
        self.assertIsInstance(expr1.args[2], NegatedStepFunction)
        self.assertIs(expr1.args[2].args[0], m.s)
        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(expr.nargs(), 3)
        self.assertIs(expr.args[0], m.p1)
        self.assertIsInstance(expr.args[1], NegatedStepFunction)
        self.assertIs(expr.args[1].args[0], m.p2)
        self.assertIs(expr.args[2], m.p1)

    def test_subtract_pulses_in_place(self):
        m = self.get_model()
        p1 = Pulse(interval_var=m.a, height=1)
        p2 = Pulse(interval_var=m.b, height=3)
        expr = p1
        expr -= p2
        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(len(expr.args), 2)
        self.assertEqual(expr.nargs(), 2)
        self.assertIs(expr.args[0], p1)
        self.assertIsInstance(expr.args[1], NegatedStepFunction)
        self.assertIs(expr.args[1].args[0], p2)

    def test_subtract_steps_in_place(self):
        m = self.get_model()
        s1 = Step(m.a.start_time, height=1)
        s2 = Step(m.b.end_time, height=3)
        expr = s1
        expr -= s2
        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(len(expr.args), 2)
        self.assertEqual(expr.nargs(), 2)
        self.assertIs(expr.args[0], s1)
        self.assertIsInstance(expr.args[1], NegatedStepFunction)
        self.assertIs(expr.args[1].args[0], s2)

    def test_subtract_from_cumul_func_in_place(self):
        m = self.get_model()
        m.p1 = Pulse(interval_var=m.a, height=5)
        m.p2 = Pulse(interval_var=m.b, height=-3)
        m.s = Step(m.b.end_time, height=5)
        expr = m.p1 + m.s
        expr -= m.p2
        self.assertIsInstance(expr, CumulativeFunction)
        self.assertEqual(expr.nargs(), 3)
        self.assertIs(expr.args[0], m.p1)
        self.assertIs(expr.args[1], m.s)
        self.assertIsInstance(expr.args[2], NegatedStepFunction)
        self.assertIs(expr.args[2].args[0], m.p2)
        self.assertEqual(str(expr), 'Pulse(a, height=5) + Step(b.end_time, height=5) - Pulse(b, height=-3)')

    def test_cannot_subtract_constant(self):
        m = self.get_model()
        with self.assertRaisesRegex(TypeError, "Cannot subtract object of class <class 'int'> from object of class <class 'pyomo.contrib.cp.scheduling_expr.step_function_expressions.StepAtStart'>"):
            expr = Step(m.a.start_time, height=6) - 3

    def test_cannot_subtract_from_constant(self):
        m = self.get_model()
        with self.assertRaisesRegex(TypeError, "Cannot subtract object of class <class 'pyomo.contrib.cp.scheduling_expr.step_function_expressions.StepAtStart'> from object of class <class 'int'>"):
            expr = 3 - Step(m.a.start_time, height=6)