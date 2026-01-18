import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestLoadVars(unittest.TestCase):

    def setUp(self):
        opt = SolverFactory('cplex', solver_io='python')
        model = ConcreteModel()
        model.X = Var(within=NonNegativeReals, initialize=0)
        model.Y = Var(within=NonNegativeReals, initialize=0)
        model.C1 = Constraint(expr=2 * model.X + model.Y >= 8)
        model.C2 = Constraint(expr=model.X + 3 * model.Y >= 6)
        model.O = Objective(expr=model.X + model.Y)
        opt.solve(model, load_solutions=False, save_results=False)
        self._model = model
        self._opt = opt

    def test_all_vars_are_loaded(self):
        self.assertTrue(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertEqual(value(self._model.X), 0)
        self.assertEqual(value(self._model.Y), 0)
        with unittest.mock.patch.object(self._opt._solver_model.solution, 'get_values', wraps=self._opt._solver_model.solution.get_values) as wrapped_values_call:
            self._opt.load_vars()
            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, tuple())
        self.assertFalse(self._model.X.stale)
        self.assertFalse(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertAlmostEqual(value(self._model.Y), 0.8)

    def test_only_specified_vars_are_loaded(self):
        self.assertTrue(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertEqual(value(self._model.X), 0)
        self.assertEqual(value(self._model.Y), 0)
        with unittest.mock.patch.object(self._opt._solver_model.solution, 'get_values', wraps=self._opt._solver_model.solution.get_values) as wrapped_values_call:
            self._opt.load_vars([self._model.X])
            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, (([0],), {}))
        self.assertFalse(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertEqual(value(self._model.Y), 0)
        with unittest.mock.patch.object(self._opt._solver_model.solution, 'get_values', wraps=self._opt._solver_model.solution.get_values) as wrapped_values_call:
            self._opt.load_vars([self._model.Y])
            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, (([1],), {}))
        self.assertFalse(self._model.X.stale)
        self.assertFalse(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertAlmostEqual(value(self._model.Y), 0.8)