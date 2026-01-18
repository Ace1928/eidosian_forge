import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
class TestTrustRegionInterface(unittest.TestCase):

    def setUp(self):
        self.m = ConcreteModel()
        self.m.z = Var(range(3), domain=Reals, initialize=2.0)
        self.m.x = Var(range(2), initialize=2.0)
        self.m.x[1] = 1.0

        def blackbox(a, b):
            return sin(a - b)

        def grad_blackbox(args, fixed):
            a, b = args[:2]
            return [cos(a - b), -cos(a - b)]
        self.m.bb = ExternalFunction(blackbox, grad_blackbox)
        self.m.obj = Objective(expr=(self.m.z[0] - 1.0) ** 2 + (self.m.z[0] - self.m.z[1]) ** 2 + (self.m.z[2] - 1.0) ** 2 + (self.m.x[0] - 1.0) ** 4 + (self.m.x[1] - 1.0) ** 6)
        self.m.c1 = Constraint(expr=self.m.x[0] * self.m.z[0] ** 2 + self.m.bb(self.m.x[0], self.m.x[1]) == 2 * sqrt(2.0))
        self.m.c2 = Constraint(expr=self.m.z[2] ** 4 * self.m.z[1] ** 2 + self.m.z[1] == 8 + sqrt(2.0))
        self.config = _trf_config()
        self.ext_fcn_surrogate_map_rule = lambda comp, ef: 0
        self.interface = TRFInterface(self.m, [self.m.z[0], self.m.z[1], self.m.z[2]], self.ext_fcn_surrogate_map_rule, self.config)

    def test_initializeInterface(self):
        self.assertEqual(self.m, self.interface.original_model)
        self.assertEqual(self.config, self.interface.config)
        self.assertEqual(self.interface.basis_expression_rule, self.ext_fcn_surrogate_map_rule)
        self.assertEqual('ipopt', self.interface.solver.name)

    def test_replaceRF(self):
        self.interface.data.all_variables = ComponentSet()
        self.interface.data.truth_models = ComponentMap()
        self.interface.data.ef_outputs = VarList()
        expr = self.interface.model.obj.expr
        new_expr = self.interface.replaceEF(expr)
        self.assertEqual(expr, new_expr)
        expr = self.interface.model.c1.expr
        new_expr = self.interface.replaceEF(expr)
        self.assertIsNot(expr, new_expr)
        self.assertEqual(str(new_expr), 'x[0]*z[0]**2 + trf_data.ef_outputs[1]  ==  2.8284271247461903')

    def test_remove_ef_from_expr(self):
        self.interface.data.all_variables = ComponentSet()
        self.interface.data.truth_models = ComponentMap()
        self.interface.data.ef_outputs = VarList()
        self.interface.data.basis_expressions = ComponentMap()
        component = self.interface.model.obj
        self.interface._remove_ef_from_expr(component)
        self.assertEqual(str(self.interface.model.obj.expr), '(z[0] - 1.0)**2 + (z[0] - z[1])**2 + (z[2] - 1.0)**2 + (x[0] - 1.0)**4 + (x[1] - 1.0)**6')
        component = self.interface.model.c1
        str_expr = str(component.expr)
        self.interface._remove_ef_from_expr(component)
        self.assertNotEqual(str_expr, str(component.expr))
        self.assertEqual(str(component.expr), 'x[0]*z[0]**2 + trf_data.ef_outputs[1]  ==  2.8284271247461903')

    def test_replaceExternalFunctionsWithVariables(self):
        self.interface.replaceExternalFunctionsWithVariables()
        for var in self.interface.model.component_data_objects(Var):
            self.assertIn(var, ComponentSet(self.interface.data.all_variables))
        for i in self.interface.data.ef_outputs:
            self.assertIn(self.interface.data.ef_outputs[i], ComponentSet(self.interface.data.all_variables))
        for i, k in self.interface.data.truth_models.items():
            self.assertIsInstance(k, ExternalFunctionExpression)
            self.assertIn(str(self.interface.model.x[0]), str(k))
            self.assertIn(str(self.interface.model.x[1]), str(k))
            self.assertIsInstance(i, _GeneralVarData)
            self.assertEqual(i, self.interface.data.ef_outputs[1])
        for i, k in self.interface.data.basis_expressions.items():
            self.assertEqual(k, 0)
            self.assertEqual(i, self.interface.data.ef_outputs[1])
        self.assertEqual(1, list(self.interface.data.ef_inputs.keys())[0])
        self.assertEqual(self.interface.data.ef_inputs[1], [self.interface.model.x[0], self.interface.model.x[1]])
        self.assertEqual(list(self.interface.model.component_objects(ExternalFunction)), [])
        self.m.obj2 = Objective(expr=self.m.x[0] ** 2 - (self.m.z[1] - 3) ** 3)
        interface = TRFInterface(self.m, [self.m.z[0], self.m.z[1], self.m.z[2]], self.ext_fcn_surrogate_map_rule, self.config)
        with self.assertRaises(ValueError):
            interface.replaceExternalFunctionsWithVariables()

    def test_createConstraints(self):
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        self.assertFalse(self.interface.data.basis_constraint.active)
        self.assertFalse(self.interface.data.sm_constraint_basis.active)
        self.assertEqual(len(self.interface.data.basis_constraint), 1)
        self.assertEqual(len(self.interface.data.sm_constraint_basis), 1)
        self.assertEqual(list(self.interface.data.basis_constraint.keys()), [1])
        cs = ComponentSet(identify_variables(self.interface.data.basis_constraint[1].expr))
        self.assertEqual(len(cs), 1)
        self.assertIn(self.interface.data.ef_outputs[1], cs)
        cs = ComponentSet(identify_variables(self.interface.data.sm_constraint_basis[1].expr))
        self.assertEqual(len(cs), 3)
        self.assertIn(self.interface.model.x[0], cs)
        self.assertIn(self.interface.model.x[1], cs)
        self.assertIn(self.interface.data.ef_outputs[1], cs)

    def test_updateSurrogateModel(self):
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        self.interface.updateSurrogateModel()
        for key, val in self.interface.data.basis_model_output.items():
            self.assertEqual(value(val), 0)
        for key, val in self.interface.data.grad_basis_model_output.items():
            self.assertEqual(value(val), 0)
        for key, val in self.interface.data.truth_model_output.items():
            self.assertEqual(value(val), 0.8414709848078965)
        truth_grads = []
        for key, val in self.interface.data.grad_truth_model_output.items():
            truth_grads.append(value(val))
        self.assertEqual(truth_grads, [cos(1), -cos(1)])
        for key, val in self.interface.data.value_of_ef_inputs.items():
            self.assertEqual(value(self.interface.model.x[key[1]]), value(val))
        self.interface.model.x.set_values({0: 0, 1: 0})
        self.interface.updateSurrogateModel()
        for key, val in self.interface.data.basis_model_output.items():
            self.assertEqual(value(val), 0)
        for key, val in self.interface.data.grad_basis_model_output.items():
            self.assertEqual(value(val), 0)
        for key, val in self.interface.data.truth_model_output.items():
            self.assertEqual(value(val), 0)
        truth_grads = []
        for key, val in self.interface.data.grad_truth_model_output.items():
            truth_grads.append(value(val))
        self.assertEqual(truth_grads, [cos(0), -cos(0)])
        for key, val in self.interface.data.value_of_ef_inputs.items():
            self.assertEqual(value(self.interface.model.x[key[1]]), value(val))

    def test_getCurrentDecisionVariableValues(self):
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        current_values = self.interface.getCurrentDecisionVariableValues()
        for var in self.interface.decision_variables:
            self.assertIn(var.name, list(current_values.keys()))
            self.assertEqual(current_values[var.name], value(var))

    @unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
    def test_updateDecisionVariableBounds(self):
        self.interface.initializeProblem()
        for var in self.interface.decision_variables:
            self.assertEqual(self.interface.initial_decision_bounds[var.name], [var.lb, var.ub])
        self.interface.updateDecisionVariableBounds(0.5)
        for var in self.interface.decision_variables:
            self.assertNotEqual(self.interface.initial_decision_bounds[var.name], [var.lb, var.ub])

    def test_getCurrentModelState(self):
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        result = self.interface.getCurrentModelState()
        self.assertEqual(len(result), len(self.interface.data.all_variables))
        for var in self.interface.data.all_variables:
            self.assertIn(value(var), result)

    @unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
    def test_calculateFeasibility(self):
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        feasibility = self.interface.calculateFeasibility()
        self.assertEqual(feasibility, 0)
        self.interface.updateSurrogateModel()
        feasibility = self.interface.calculateFeasibility()
        self.assertEqual(feasibility, 0)
        self.interface.data.basis_constraint.activate()
        objective, step_norm, feasibility = self.interface.solveModel()
        self.assertEqual(feasibility, 0.09569982275514467)
        self.interface.data.basis_constraint.deactivate()

    @unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
    def test_calculateStepSizeInfNorm(self):
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        original_values = self.interface.getCurrentDecisionVariableValues()
        self.interface.updateSurrogateModel()
        new_values = self.interface.getCurrentDecisionVariableValues()
        stepnorm = self.interface.calculateStepSizeInfNorm(original_values, new_values)
        self.assertEqual(stepnorm, 0)
        self.interface.data.basis_constraint.activate()
        objective, step_norm, feasibility = self.interface.solveModel()
        self.assertEqual(step_norm, 3.393437471478297)
        self.interface.data.basis_constraint.deactivate()

    @unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
    def test_solveModel(self):
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        self.interface.data.basis_constraint.activate()
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.data.basis_model_output[:] = 0
        self.interface.data.grad_basis_model_output[...] = 0
        self.interface.data.truth_model_output[:] = 0
        self.interface.data.grad_truth_model_output[...] = 0
        self.interface.data.value_of_ef_inputs[...] = 0
        objective, step_norm, feasibility = self.interface.solveModel()
        self.assertAlmostEqual(objective, 5.150744273013601)
        self.assertAlmostEqual(step_norm, 3.393437471478297)
        self.assertAlmostEqual(feasibility, 0.09569982275514467)
        self.interface.data.basis_constraint.deactivate()
        self.interface.updateSurrogateModel()
        self.interface.data.sm_constraint_basis.activate()
        objective, step_norm, feasibility = self.interface.solveModel()
        self.assertAlmostEqual(objective, 5.15065981284333)
        self.assertAlmostEqual(step_norm, 0.0017225116628372117)
        self.assertAlmostEqual(feasibility, 0.00014665023773349772)

    @unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
    def test_initializeProblem(self):
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        objective, feasibility = self.interface.initializeProblem()
        for var in self.interface.decision_variables:
            self.assertIn(var.name, list(self.interface.initial_decision_bounds.keys()))
            self.assertEqual(self.interface.initial_decision_bounds[var.name], [var.lb, var.ub])
        self.assertAlmostEqual(objective, 5.150744273013601)
        self.assertAlmostEqual(feasibility, 0.09569982275514467)
        self.assertTrue(self.interface.data.sm_constraint_basis.active)
        self.assertFalse(self.interface.data.basis_constraint.active)

    @unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
    def test_rejectStep(self):
        self.interface.model.x[1] = 1.5
        self.interface.model.x[0] = 2.0
        self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
        self.interface.replaceExternalFunctionsWithVariables()
        self.interface.createConstraints()
        self.interface.data.basis_constraint.activate()
        _, _, _ = self.interface.solveModel()
        self.assertEqual(len(self.interface.data.all_variables), len(self.interface.data.previous_model_state))
        self.assertNotEqual(value(self.interface.model.x[0]), 2.0)
        self.assertNotEqual(value(self.interface.model.x[1]), 1.5)
        self.assertNotEqual(value(self.interface.model.z[0]), 5.0)
        self.assertNotEqual(value(self.interface.model.z[1]), 2.5)
        self.assertNotEqual(value(self.interface.model.z[2]), -1.0)
        self.interface.rejectStep()
        self.assertEqual(value(self.interface.model.x[0]), 2.0)
        self.assertEqual(value(self.interface.model.x[1]), 1.5)
        self.assertEqual(value(self.interface.model.z[0]), 5.0)
        self.assertEqual(value(self.interface.model.z[1]), 2.5)
        self.assertEqual(value(self.interface.model.z[2]), -1.0)