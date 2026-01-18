import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
class TestTemporarySubsystemManager(unittest.TestCase):

    def test_context(self):
        m = _make_simple_model()
        to_fix = [m.v4]
        to_deactivate = [m.con1]
        to_reset = [m.v1]
        m.v1.set_value(1.5)
        with TemporarySubsystemManager(to_fix, to_deactivate, to_reset):
            self.assertEqual(m.v1.value, 1.5)
            self.assertTrue(m.v4.fixed)
            self.assertFalse(m.con1.active)
            m.v1.set_value(2.0)
            m.v4.set_value(3.0)
        self.assertEqual(m.v1.value, 1.5)
        self.assertEqual(m.v4.value, 3.0)
        self.assertFalse(m.v4.fixed)
        self.assertTrue(m.con1.active)

    def test_context_some_redundant(self):
        m = _make_simple_model()
        to_fix = [m.v2, m.v4]
        to_deactivate = [m.con1, m.con2]
        to_reset = [m.v1]
        m.v1.set_value(1.5)
        m.v2.fix()
        m.con1.deactivate()
        with TemporarySubsystemManager(to_fix, to_deactivate, to_reset):
            self.assertEqual(m.v1.value, 1.5)
            self.assertTrue(m.v2.fixed)
            self.assertTrue(m.v4.fixed)
            self.assertFalse(m.con1.active)
            self.assertFalse(m.con2.active)
            m.v1.set_value(2.0)
            m.v2.set_value(3.0)
        self.assertEqual(m.v1.value, 1.5)
        self.assertEqual(m.v2.value, 3.0)
        self.assertTrue(m.v2.fixed)
        self.assertFalse(m.v4.fixed)
        self.assertTrue(m.con2.active)
        self.assertFalse(m.con1.active)

    @unittest.skipUnless(pyo.SolverFactory('ipopt').available(), 'Ipopt is not available')
    def test_fix_then_solve(self):
        m = _make_simple_model()
        ipopt = pyo.SolverFactory('ipopt')
        m.v1.set_value(1.0)
        m.v2.set_value(1.0)
        m.v3.set_value(1.0)
        m.v4.set_value(2.0)
        with TemporarySubsystemManager(to_fix=[m.v3, m.v4], to_deactivate=[m.con1]):
            ipopt.solve(m)
        self.assertAlmostEqual(m.v1.value, pyo.sqrt(7.0), delta=1e-08)
        self.assertAlmostEqual(m.v2.value, pyo.sqrt(4.0 - pyo.sqrt(7.0)), delta=1e-08)

    def test_generate_subsystems_with_exception(self):
        m = _make_simple_model()
        subsystems = [([m.con1], [m.v1, m.v4]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1, m.v4]]
        block = create_subsystem_block(*subsystems[0])
        with self.assertRaises(RuntimeError):
            inputs = list(block.input_vars[:])
            with TemporarySubsystemManager(to_fix=inputs):
                self.assertTrue(all((var.fixed for var in inputs)))
                self.assertFalse(any((var.fixed for var in block.vars[:])))
                raise RuntimeError()
        self.assertFalse(any((var.fixed for var in m.component_data_objects(pyo.Var))))

    def test_to_unfix(self):
        m = _make_simple_model()
        m.v1.fix()
        m.v3.fix()
        with TemporarySubsystemManager(to_unfix=[m.v3]):
            self.assertTrue(m.v1.fixed)
            self.assertFalse(m.v2.fixed)
            self.assertFalse(m.v3.fixed)
            self.assertFalse(m.v4.fixed)
        self.assertTrue(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertFalse(m.v4.fixed)