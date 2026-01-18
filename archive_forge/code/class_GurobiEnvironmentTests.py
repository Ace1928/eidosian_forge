import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
@unittest.skipIf(not gurobi_available, 'gurobi license is not valid')
class GurobiEnvironmentTests(GurobiBase):

    def assert_optimal_result(self, results):
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)

    def test_init_default_env(self):
        with patch('gurobipy.Model') as PatchModel:
            with SolverFactory('gurobi_direct') as opt:
                opt.available()
                opt.available()
                PatchModel.assert_called_once_with()

    def test_close_global(self):
        with patch('gurobipy.Model') as PatchModel, patch('gurobipy.disposeDefaultEnv') as patch_dispose:
            with SolverFactory('gurobi_direct') as opt:
                opt.available()
                opt.available()
                PatchModel.assert_called_once_with()
            patch_dispose.assert_not_called()
            opt.close_global()
            patch_dispose.assert_called_once_with()
        with patch('gurobipy.Model') as PatchModel, patch('gurobipy.disposeDefaultEnv') as patch_dispose:
            with SolverFactory('gurobi_direct') as opt:
                opt.available()
                opt.available()
                PatchModel.assert_called_once_with()
            patch_dispose.assert_not_called()

    def test_persisted_license_failure(self):
        with patch('gurobipy.Model', side_effect=gp.GurobiError(NO_LICENSE, 'nolicense')):
            with SolverFactory('gurobi_direct') as opt:
                with self.assertRaisesRegex(ApplicationError, 'nolicense'):
                    opt.solve(self.model)
        with SolverFactory('gurobi_direct') as opt:
            results = opt.solve(self.model)
            self.assert_optimal_result(results)

    def test_persisted_license_failure_managed(self):
        with patch('gurobipy.Env', side_effect=gp.GurobiError(NO_LICENSE, 'nolicense')):
            with SolverFactory('gurobi_direct', manage_env=True) as opt:
                with self.assertRaisesRegex(ApplicationError, 'nolicense'):
                    opt.solve(self.model)
        with SolverFactory('gurobi_direct', manage_env=True) as opt:
            results = opt.solve(self.model)
            self.assert_optimal_result(results)
            self.assertEqual(results.solver.status, SolverStatus.ok)

    def test_context(self):
        with gp.Env() as use_env:
            with patch('gurobipy.Env', return_value=use_env):
                with SolverFactory('gurobi_direct', manage_env=True) as opt:
                    results = opt.solve(self.model)
                    self.assert_optimal_result(results)
            with self.assertRaises(gp.GurobiError):
                use_env.start()

    def test_close(self):
        with gp.Env() as use_env:
            with patch('gurobipy.Env', return_value=use_env):
                opt = SolverFactory('gurobi_direct', manage_env=True)
                try:
                    results = opt.solve(self.model)
                    self.assert_optimal_result(results)
                finally:
                    opt.close()
            with self.assertRaises(gp.GurobiError):
                use_env.start()

    @unittest.skipIf(single_use_license(), reason='test requires multi-use license')
    def test_multiple_solvers_managed(self):
        with SolverFactory('gurobi_direct', manage_env=True) as opt1, SolverFactory('gurobi_direct', manage_env=True) as opt2:
            results1 = opt1.solve(self.model)
            self.assert_optimal_result(results1)
            results2 = opt2.solve(self.model)
            self.assert_optimal_result(results2)

    def test_multiple_solvers_nonmanaged(self):
        with SolverFactory('gurobi_direct') as opt1, SolverFactory('gurobi_direct') as opt2:
            results1 = opt1.solve(self.model)
            self.assert_optimal_result(results1)
            results2 = opt2.solve(self.model)
            self.assert_optimal_result(results2)

    @unittest.skipIf(single_use_license(), reason='test requires multi-use license')
    def test_managed_env(self):
        gp.setParam('IterationLimit', 100)
        with gp.Env(params={'IterationLimit': 0, 'Presolve': 0}) as use_env, patch('gurobipy.Env', return_value=use_env):
            with SolverFactory('gurobi_direct', manage_env=True) as opt:
                results = opt.solve(self.model)
                self.assertEqual(results.solver.status, SolverStatus.aborted)
                self.assertEqual(results.solver.termination_condition, TerminationCondition.maxIterations)

    def test_nonmanaged_env(self):
        gp.setParam('IterationLimit', 0)
        gp.setParam('Presolve', 0)
        with SolverFactory('gurobi_direct') as opt:
            results = opt.solve(self.model)
            self.assertEqual(results.solver.status, SolverStatus.aborted)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.maxIterations)