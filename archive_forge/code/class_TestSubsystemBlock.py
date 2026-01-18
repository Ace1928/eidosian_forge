import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
class TestSubsystemBlock(unittest.TestCase):

    def test_square_subsystem(self):
        m = _make_simple_model()
        cons = [m.con2, m.con3]
        vars = [m.v1, m.v2]
        block = create_subsystem_block(cons, vars)
        self.assertEqual(len(block.vars), 2)
        self.assertEqual(len(block.cons), 2)
        self.assertEqual(len(block.input_vars), 2)
        self.assertEqual(len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 4)
        block.input_vars.fix()
        self.assertEqual(len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 2)
        self.assertIs(block.cons[0], m.con2)
        self.assertIs(block.cons[1], m.con3)
        self.assertIs(block.vars[0], m.v1)
        self.assertIs(block.vars[1], m.v2)
        self.assertIs(block.input_vars[0], m.v4)
        self.assertIs(block.input_vars[1], m.v3)
        self.assertIsNot(block.model(), m)
        for comp in block.component_objects((pyo.Var, pyo.Constraint)):
            self.assertTrue(comp.is_reference())
            for data in comp.values():
                self.assertIs(data.model(), m)

    def test_subsystem_inputs_only(self):
        m = _make_simple_model()
        cons = [m.con2, m.con3]
        block = create_subsystem_block(cons)
        self.assertEqual(len(block.vars), 0)
        self.assertEqual(len(block.input_vars), 4)
        self.assertEqual(len(block.cons), 2)
        self.assertEqual(len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 4)
        block.input_vars.fix()
        self.assertEqual(len([v for v in block.component_data_objects(pyo.Var) if not v.fixed]), 0)
        var_set = ComponentSet([m.v1, m.v2, m.v3, m.v4])
        self.assertIs(block.cons[0], m.con2)
        self.assertIs(block.cons[1], m.con3)
        self.assertIn(block.input_vars[0], var_set)
        self.assertIn(block.input_vars[1], var_set)
        self.assertIn(block.input_vars[2], var_set)
        self.assertIn(block.input_vars[3], var_set)
        self.assertIsNot(block.model(), m)
        for comp in block.component_objects((pyo.Var, pyo.Constraint)):
            self.assertTrue(comp.is_reference())
            for data in comp.values():
                self.assertIs(data.model(), m)

    @unittest.skipUnless(pyo.SolverFactory('ipopt').available(), 'Ipopt is not available')
    def test_solve_subsystem(self):
        m = _make_simple_model()
        ipopt = pyo.SolverFactory('ipopt')
        m.v5 = pyo.Var(initialize=1.0)
        m.c4 = pyo.Constraint(expr=m.v5 == 5.0)
        cons = [m.con2, m.con3]
        vars = [m.v1, m.v2]
        block = create_subsystem_block(cons, vars)
        m.v3.fix(1.0)
        m.v4.fix(2.0)
        m.v1.set_value(1.0)
        m.v2.set_value(1.0)
        ipopt.solve(block)
        self.assertAlmostEqual(m.v1.value, pyo.sqrt(7.0), delta=1e-08)
        self.assertAlmostEqual(m.v2.value, pyo.sqrt(4.0 - pyo.sqrt(7.0)), delta=1e-08)
        self.assertEqual(m.v5.value, 1.0)

    def test_generate_subsystems_without_fixed_var(self):
        m = _make_simple_model()
        subs = [([m.con1], [m.v1, m.v4]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1, m.v4]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
            with TemporarySubsystemManager(to_fix=inputs):
                self.assertIs(block.model(), block)
                var_set = ComponentSet(subs[i][1])
                con_set = ComponentSet(subs[i][0])
                input_set = ComponentSet(other_vars[i])
                self.assertEqual(len(var_set), len(block.vars))
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(input_set), len(block.input_vars))
                self.assertTrue(all((var in var_set for var in block.vars[:])))
                self.assertTrue(all((con in con_set for con in block.cons[:])))
                self.assertTrue(all((var in input_set for var in inputs)))
                self.assertTrue(all((var.fixed for var in inputs)))
                self.assertFalse(any((var.fixed for var in block.vars[:])))
        self.assertFalse(any((var.fixed for var in m.component_data_objects(pyo.Var))))

    def test_generate_subsystems_with_exception(self):
        m = _make_simple_model()
        subs = [([m.con1], [m.v1, m.v4]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1, m.v4]]
        with self.assertRaises(RuntimeError):
            for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
                with TemporarySubsystemManager(to_fix=inputs):
                    self.assertTrue(all((var.fixed for var in inputs)))
                    self.assertFalse(any((var.fixed for var in block.vars[:])))
                    if i == 1:
                        raise RuntimeError()
        self.assertFalse(any((var.fixed for var in m.component_data_objects(pyo.Var))))

    def test_generate_subsystems_with_fixed_var(self):
        m = _make_simple_model()
        m.v4.fix()
        subs = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
            inputs = list(block.input_vars.values())
            with TemporarySubsystemManager(to_fix=inputs):
                self.assertIs(block.model(), block)
                var_set = ComponentSet(subs[i][1])
                con_set = ComponentSet(subs[i][0])
                input_set = ComponentSet(other_vars[i])
                self.assertEqual(len(var_set), len(block.vars))
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(input_set), len(inputs))
                self.assertTrue(all((var in var_set for var in block.vars[:])))
                self.assertTrue(all((con in con_set for con in block.cons[:])))
                self.assertTrue(all((var in input_set for var in inputs)))
                self.assertTrue(all((var.fixed for var in inputs)))
                self.assertFalse(any((var.fixed for var in block.vars[:])))
        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertTrue(m.v4.fixed)

    def test_generate_subsystems_include_fixed_var(self):
        m = _make_simple_model()
        m.v4.fix()
        subsystems = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3, m.v4], [m.v1, m.v4]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subsystems, include_fixed=True)):
            with TemporarySubsystemManager(to_fix=inputs):
                self.assertIs(block.model(), block)
                var_set = ComponentSet(subsystems[i][1])
                con_set = ComponentSet(subsystems[i][0])
                input_set = ComponentSet(other_vars[i])
                self.assertEqual(len(var_set), len(block.vars))
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(input_set), len(block.input_vars))
                self.assertTrue(all((var in var_set for var in block.vars[:])))
                self.assertTrue(all((con in con_set for con in block.cons[:])))
                self.assertTrue(all((var in input_set for var in inputs)))
                self.assertTrue(all((var.fixed for var in inputs)))
                self.assertFalse(any((var.fixed for var in block.vars[:])))
        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertTrue(m.v4.fixed)

    def test_generate_subsystems_dont_fix_inputs(self):
        m = _make_simple_model()
        subs = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3, m.v4], [m.v1, m.v4]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
            self.assertIs(block.model(), block)
            var_set = ComponentSet(subs[i][1])
            con_set = ComponentSet(subs[i][0])
            input_set = ComponentSet(other_vars[i])
            self.assertEqual(len(var_set), len(block.vars))
            self.assertEqual(len(con_set), len(block.cons))
            self.assertEqual(len(input_set), len(inputs))
            self.assertTrue(all((var in var_set for var in block.vars[:])))
            self.assertTrue(all((con in con_set for con in block.cons[:])))
            self.assertTrue(all((var in input_set for var in inputs)))
            self.assertFalse(any((var.fixed for var in inputs)))
            self.assertFalse(any((var.fixed for var in block.vars[:])))
        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertFalse(m.v4.fixed)

    def test_generate_dont_fix_inputs_with_fixed_var(self):
        m = _make_simple_model()
        m.v4.fix()
        subs = [([m.con1], [m.v1]), ([m.con2, m.con3], [m.v2, m.v3])]
        other_vars = [[m.v2, m.v3], [m.v1]]
        for i, (block, inputs) in enumerate(generate_subsystem_blocks(subs)):
            self.assertIs(block.model(), block)
            var_set = ComponentSet(subs[i][1])
            con_set = ComponentSet(subs[i][0])
            input_set = ComponentSet(other_vars[i])
            self.assertEqual(len(var_set), len(block.vars))
            self.assertEqual(len(con_set), len(block.cons))
            self.assertEqual(len(input_set), len(inputs))
            self.assertTrue(all((var in var_set for var in block.vars[:])))
            self.assertTrue(all((con in con_set for con in block.cons[:])))
            self.assertTrue(all((var in input_set for var in inputs)))
            self.assertFalse(m.v1.fixed)
            self.assertFalse(m.v2.fixed)
            self.assertFalse(m.v3.fixed)
            self.assertTrue(m.v4.fixed)
        self.assertFalse(m.v1.fixed)
        self.assertFalse(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertTrue(m.v4.fixed)

    def _make_model_with_external_functions(self):
        m = pyo.ConcreteModel()
        gsl = find_GSL()
        m.bessel = pyo.ExternalFunction(library=gsl, function='gsl_sf_bessel_J0')
        m.fermi = pyo.ExternalFunction(library=gsl, function='gsl_sf_fermi_dirac_m1')
        m.v1 = pyo.Var(initialize=1.0)
        m.v2 = pyo.Var(initialize=2.0)
        m.v3 = pyo.Var(initialize=3.0)
        m.con1 = pyo.Constraint(expr=m.v1 == 0.5)
        m.con2 = pyo.Constraint(expr=2 * m.fermi(m.v1) + m.v2 ** 2 - m.v3 == 1.0)
        m.con3 = pyo.Constraint(expr=m.bessel(m.v1) - m.bessel(m.v2) + m.v3 ** 2 == 2.0)
        return m

    @unittest.skipUnless(find_GSL(), 'Could not find the AMPL GSL library')
    def test_identify_external_functions(self):
        m = self._make_model_with_external_functions()
        m._con = pyo.Constraint(expr=2 * m.fermi(m.bessel(m.v1 ** 2) + 0.1) == 1.0)
        gsl = find_GSL()
        fcns = list(identify_external_functions(m.con2.expr))
        self.assertEqual(len(fcns), 1)
        self.assertEqual(fcns[0]._fcn._library, gsl)
        self.assertEqual(fcns[0]._fcn._function, 'gsl_sf_fermi_dirac_m1')
        fcns = list(identify_external_functions(m.con3.expr))
        fcn_data = set(((fcn._fcn._library, fcn._fcn._function) for fcn in fcns))
        self.assertEqual(len(fcns), 2)
        pred_fcn_data = {(gsl, 'gsl_sf_bessel_J0')}
        self.assertEqual(fcn_data, pred_fcn_data)
        fcns = list(identify_external_functions(m._con.expr))
        fcn_data = set(((fcn._fcn._library, fcn._fcn._function) for fcn in fcns))
        self.assertEqual(len(fcns), 2)
        pred_fcn_data = {(gsl, 'gsl_sf_bessel_J0'), (gsl, 'gsl_sf_fermi_dirac_m1')}
        self.assertEqual(fcn_data, pred_fcn_data)

    def _solve_ef_model_with_ipopt(self):
        m = self._make_model_with_external_functions()
        ipopt = pyo.SolverFactory('ipopt')
        ipopt.solve(m)
        return m

    @unittest.skipUnless(find_GSL(), 'Could not find the AMPL GSL library')
    @unittest.skipUnless(pyo.SolverFactory('ipopt').available(), 'ipopt is not available')
    def test_with_external_function(self):
        m = self._make_model_with_external_functions()
        subsystem = ([m.con2, m.con3], [m.v2, m.v3])
        m.v1.set_value(0.5)
        block = create_subsystem_block(*subsystem)
        ipopt = pyo.SolverFactory('ipopt')
        with TemporarySubsystemManager(to_fix=list(block.input_vars.values())):
            ipopt.solve(block)
        self.assertEqual(m.v1.value, 0.5)
        self.assertFalse(m.v1.fixed)
        self.assertAlmostEqual(m.v2.value, 1.04816, delta=1e-05)
        self.assertAlmostEqual(m.v3.value, 1.34356, delta=1e-05)
        m_full = self._solve_ef_model_with_ipopt()
        self.assertAlmostEqual(m.v1.value, m_full.v1.value)
        self.assertAlmostEqual(m.v2.value, m_full.v2.value)
        self.assertAlmostEqual(m.v3.value, m_full.v3.value)

    @unittest.skipUnless(find_GSL(), 'Could not find the AMPL GSL library')
    def test_external_function_with_potential_name_collision(self):
        m = self._make_model_with_external_functions()
        m.b = pyo.Block()
        m.b._gsl_sf_bessel_J0 = pyo.Var()
        m.b.con = pyo.Constraint(expr=m.b._gsl_sf_bessel_J0 == m.bessel(m.v1))
        add_local_external_functions(m.b)
        self.assertTrue(isinstance(m.b._gsl_sf_bessel_J0, pyo.Var))
        ex_fcns = list(m.b.component_objects(pyo.ExternalFunction))
        self.assertEqual(len(ex_fcns), 1)
        fcn = ex_fcns[0]
        self.assertEqual(fcn._function, 'gsl_sf_bessel_J0')