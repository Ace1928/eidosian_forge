import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
@unittest.skipUnless(scipy_available, 'SciPy is not available')
@unittest.skipUnless(networkx_available, 'NetworkX is not available')
class TestGenerateSCC(unittest.TestCase):

    def test_gas_expansion(self):
        N = 5
        m = make_gas_expansion_model(N)
        m.rho[0].fix()
        m.F[0].fix()
        m.T[0].fix()
        constraints = list(m.component_data_objects(pyo.Constraint))
        self.assertEqual(len(list(generate_strongly_connected_components(constraints))), N + 1)
        for i, (block, inputs) in enumerate(generate_strongly_connected_components(constraints)):
            with TemporarySubsystemManager(to_fix=inputs):
                if i == 0:
                    self.assertEqual(len(block.vars), 1)
                    self.assertEqual(len(block.cons), 1)
                    var_set = ComponentSet([m.P[i]])
                    con_set = ComponentSet([m.ideal_gas[i]])
                    for var, con in zip(block.vars[:], block.cons[:]):
                        self.assertIn(var, var_set)
                        self.assertIn(con, con_set)
                    self.assertEqual(len(block.input_vars), 0)
                elif i == 1:
                    self.assertEqual(len(block.vars), 4)
                    self.assertEqual(len(block.cons), 4)
                    var_set = ComponentSet([m.P[i], m.rho[i], m.F[i], m.T[i]])
                    con_set = ComponentSet([m.ideal_gas[i], m.mbal[i], m.ebal[i], m.expansion[i]])
                    for var, con in zip(block.vars[:], block.cons[:]):
                        self.assertIn(var, var_set)
                        self.assertIn(con, con_set)
                    other_var_set = ComponentSet([m.P[i - 1]])
                    self.assertEqual(len(block.input_vars), 1)
                    for var in block.input_vars[:]:
                        self.assertIn(var, other_var_set)
                else:
                    self.assertEqual(len(block.vars), 4)
                    self.assertEqual(len(block.cons), 4)
                    var_set = ComponentSet([m.P[i], m.rho[i], m.F[i], m.T[i]])
                    con_set = ComponentSet([m.ideal_gas[i], m.mbal[i], m.ebal[i], m.expansion[i]])
                    for var, con in zip(block.vars[:], block.cons[:]):
                        self.assertIn(var, var_set)
                        self.assertIn(con, con_set)
                    other_var_set = ComponentSet([m.P[i - 1], m.rho[i - 1], m.F[i - 1], m.T[i - 1]])
                    self.assertEqual(len(block.input_vars), 4)
                    for var in block.input_vars[:]:
                        self.assertIn(var, other_var_set)

    def test_dynamic_backward_disc_with_initial_conditions(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme='BACKWARD')
        time = m.time
        t0 = m.time.first()
        t1 = m.time.next(t0)
        m.flow_in.fix()
        m.height[t0].fix()
        constraints = list(m.component_data_objects(pyo.Constraint))
        self.assertEqual(len(list(generate_strongly_connected_components(constraints))), nfe + 2)
        t_scc_map = {}
        for i, (block, inputs) in enumerate(generate_strongly_connected_components(constraints)):
            with TemporarySubsystemManager(to_fix=inputs):
                t = block.vars[0].index()
                t_scc_map[t] = i
                if t == t0:
                    continue
                else:
                    t_prev = m.time.prev(t)
                    con_set = ComponentSet([m.diff_eqn[t], m.flow_out_eqn[t], m.dhdt_disc_eq[t]])
                    var_set = ComponentSet([m.height[t], m.dhdt[t], m.flow_out[t]])
                    self.assertEqual(len(con_set), len(block.cons))
                    self.assertEqual(len(var_set), len(block.vars))
                    for var, con in zip(block.vars[:], block.cons[:]):
                        self.assertIn(var, var_set)
                        self.assertIn(con, con_set)
                        self.assertFalse(var.fixed)
                    other_var_set = ComponentSet([m.height[t_prev]]) if t != t1 else ComponentSet()
                    self.assertEqual(len(inputs), len(other_var_set))
                    for var in block.input_vars[:]:
                        self.assertIn(var, other_var_set)
                        self.assertTrue(var.fixed)
        scc = -1
        for t in m.time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
                self.assertGreater(t_scc_map[t], scc)
                scc = t_scc_map[t]
            self.assertFalse(m.flow_out[t].fixed)
            self.assertFalse(m.dhdt[t].fixed)
            self.assertTrue(m.flow_in[t].fixed)

    def test_dynamic_backward_disc_without_initial_conditions(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme='BACKWARD')
        time = m.time
        t0 = m.time.first()
        t1 = m.time.next(t0)
        m.flow_in.fix()
        m.height[t0].fix()
        m.flow_out[t0].fix()
        m.dhdt[t0].fix()
        m.diff_eqn[t0].deactivate()
        m.flow_out_eqn[t0].deactivate()
        constraints = list(m.component_data_objects(pyo.Constraint, active=True))
        self.assertEqual(len(list(generate_strongly_connected_components(constraints))), nfe)
        for i, (block, inputs) in enumerate(generate_strongly_connected_components(constraints)):
            with TemporarySubsystemManager(to_fix=inputs):
                t = m.time[i + 2]
                t_prev = m.time.prev(t)
                con_set = ComponentSet([m.diff_eqn[t], m.flow_out_eqn[t], m.dhdt_disc_eq[t]])
                var_set = ComponentSet([m.height[t], m.dhdt[t], m.flow_out[t]])
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(var_set), len(block.vars))
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)
                    self.assertFalse(var.fixed)
                other_var_set = ComponentSet([m.height[t_prev]]) if t != t1 else ComponentSet()
                self.assertEqual(len(inputs), len(other_var_set))
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)
                    self.assertTrue(var.fixed)
        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
                self.assertTrue(m.flow_out[t].fixed)
                self.assertTrue(m.dhdt[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
                self.assertFalse(m.flow_out[t].fixed)
                self.assertFalse(m.dhdt[t].fixed)

    def test_dynamic_backward_with_inputs(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme='BACKWARD')
        time = m.time
        t0 = m.time.first()
        t1 = m.time.next(t0)
        m.height[t0].fix()
        m.flow_out[t0].fix()
        m.dhdt[t0].fix()
        m.diff_eqn[t0].deactivate()
        m.flow_out_eqn[t0].deactivate()
        variables = [var for var in m.component_data_objects(pyo.Var) if not var.fixed and var.parent_component() is not m.flow_in]
        constraints = list(m.component_data_objects(pyo.Constraint, active=True))
        self.assertEqual(len(list(generate_strongly_connected_components(constraints, variables))), nfe)
        for i, (block, inputs) in enumerate(generate_strongly_connected_components(constraints, variables)):
            with TemporarySubsystemManager(to_fix=inputs):
                t = m.time[i + 2]
                t_prev = m.time.prev(t)
                con_set = ComponentSet([m.diff_eqn[t], m.flow_out_eqn[t], m.dhdt_disc_eq[t]])
                var_set = ComponentSet([m.height[t], m.dhdt[t], m.flow_out[t]])
                self.assertEqual(len(con_set), len(block.cons))
                self.assertEqual(len(var_set), len(block.vars))
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)
                    self.assertFalse(var.fixed)
                other_var_set = ComponentSet([m.flow_in[t]])
                if t != t1:
                    other_var_set.add(m.height[t_prev])
                self.assertEqual(len(inputs), len(other_var_set))
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)
                    self.assertTrue(var.fixed)
        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
                self.assertTrue(m.flow_out[t].fixed)
                self.assertTrue(m.dhdt[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
                self.assertFalse(m.flow_out[t].fixed)
                self.assertFalse(m.dhdt[t].fixed)

    def test_dynamic_forward_disc(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme='FORWARD')
        time = m.time
        t0 = m.time.first()
        t1 = m.time.next(t0)
        m.flow_in.fix()
        m.height[t0].fix()
        constraints = list(m.component_data_objects(pyo.Constraint))
        self.assertEqual(len(list(generate_strongly_connected_components(constraints))), len(list(m.component_data_objects(pyo.Constraint))))
        self.assertEqual(len(list(generate_strongly_connected_components(constraints))), 3 * nfe + 2)
        for i, (block, inputs) in enumerate(generate_strongly_connected_components(constraints)):
            with TemporarySubsystemManager(to_fix=inputs):
                idx = i // 3
                mod = i % 3
                t = m.time[idx + 1]
                if t != time.last():
                    t_next = m.time.next(t)
                self.assertEqual(len(block.vars), 1)
                self.assertEqual(len(block.cons), 1)
                if mod == 0:
                    self.assertIs(block.vars[0], m.flow_out[t])
                    self.assertIs(block.cons[0], m.flow_out_eqn[t])
                elif mod == 1:
                    self.assertIs(block.vars[0], m.dhdt[t])
                    self.assertIs(block.cons[0], m.diff_eqn[t])
                elif mod == 2:
                    self.assertIs(block.vars[0], m.height[t_next])
                    self.assertIs(block.cons[0], m.dhdt_disc_eq[t])

    def test_with_zero_coefficients(self):
        """Test where the blocks we identify are incorrect if we don't filter
        out variables with coefficients of zero
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        m.eq1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + 0 * m.x[3] == 7)
        m.eq2 = pyo.Constraint(expr=m.x[1] + pyo.log(m.x[1]) == 0)
        blocks = generate_strongly_connected_components([m.eq1, m.eq2])
        blocks = [bl for bl, _ in blocks]
        self.assertEqual(len(blocks[0].vars), 1)
        self.assertIs(blocks[0].vars[0], m.x[1])
        self.assertIs(blocks[0].cons[0], m.eq2)
        self.assertEqual(len(blocks[1].vars), 1)
        self.assertIs(blocks[1].vars[0], m.x[2])
        self.assertIs(blocks[1].cons[0], m.eq1)