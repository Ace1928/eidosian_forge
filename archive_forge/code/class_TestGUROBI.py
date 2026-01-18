import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
@unittest.skipUnless('GUROBI' in INSTALLED_SOLVERS, 'GUROBI is not installed.')
class TestGUROBI(BaseTest):
    """NOTE: solves of LPs (or MILPs) get routed through GUROBI's QP interface!
    So many of these tests are testing the behavior of qurobi_qpif.py"""

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')
        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_gurobi_warm_start(self) -> None:
        """Make sure that warm starting Gurobi behaves as expected
           Note: This only checks output, not whether or not Gurobi is warm starting internally
        """
        if cp.GUROBI in INSTALLED_SOLVERS:
            import gurobipy
            import numpy as np
            A = cp.Parameter((2, 2))
            b = cp.Parameter(2)
            h = cp.Parameter(2)
            c = cp.Parameter(2)
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])
            objective = cp.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] ** 2 <= h[0] ** 2, self.x[1] <= h[1], A @ self.x == b]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertAlmostEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])
            A.value = np.array([[0, 0], [0, 1]])
            b.value = np.array([0, 1])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertAlmostEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])
            c.value = np.array([1, 1])
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertAlmostEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])
            init_value = np.array([2, 3])
            self.x.value = init_value
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])
            model = prob.solver_stats.extra_stats
            model_x = model.getVars()
            for i in range(self.x.size):
                assert init_value[i] == model_x[i].start
                assert np.isclose(self.x.value[i], model_x[i].x)
            z = cp.Variable()
            Y = cp.Variable((3, 2))
            Y_val = np.reshape(np.arange(6), (3, 2))
            Y.value = Y_val + 1
            objective = cp.Maximize(z + cp.sum(Y))
            constraints = [Y <= Y_val, z <= 2]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, Y_val.sum() + 2)
            self.assertAlmostEqual(z.value, 2)
            self.assertItemsAlmostEqual(Y.value, Y_val)
            model = prob.solver_stats.extra_stats
            model_x = model.getVars()
            assert gurobipy.GRB.UNDEFINED == model_x[0].start
            assert np.isclose(2, model_x[0].x)
            for i in range(1, Y.size + 1):
                row = (i - 1) % Y.shape[0]
                col = (i - 1) // Y.shape[0]
                assert Y_val[row, col] + 1 == model_x[i].start
                assert np.isclose(Y.value[row, col], model_x[i].x)
        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % cp.GUROBI)

    def test_gurobi_time_limit_no_solution(self) -> None:
        """Make sure that if Gurobi terminates due to a time limit before finding a solution:
            1) no error is raised,
            2) solver stats are returned.
            The test is skipped if something changes on Gurobi's side so that:
            - a solution is found despite a time limit of zero,
            - a different termination criteria is hit first.
        """
        if cp.GUROBI in INSTALLED_SOLVERS:
            import gurobipy
            objective = cp.Minimize(self.x[0])
            constraints = [cp.square(self.x[0]) <= 1]
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.GUROBI, TimeLimit=0.0)
            except Exception as e:
                self.fail('An exception %s is raised instead of returning a result.' % e)
            extra_stats = None
            solver_stats = getattr(prob, 'solver_stats', None)
            if solver_stats:
                extra_stats = getattr(solver_stats, 'extra_stats', None)
            self.assertTrue(extra_stats, 'Solver stats have not been returned.')
            nb_solutions = getattr(extra_stats, 'SolCount', None)
            if nb_solutions:
                self.skipTest('Gurobi has found a solution, the test is not relevant anymore.')
            solver_status = getattr(extra_stats, 'Status', None)
            if solver_status != gurobipy.StatusConstClass.TIME_LIMIT:
                self.skipTest('Gurobi terminated for a different reason than reaching time limit, the test is not relevant anymore.')
        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.GUROBI, TimeLimit=0)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % cp.GUROBI)

    def test_gurobi_environment(self) -> None:
        """Tests that Gurobi environments can be passed to Model.
        Gurobi environments can include licensing and model parameter data.
        """
        if cp.GUROBI in INSTALLED_SOLVERS:
            import gurobipy
            params = {'MIPGap': np.random.random(), 'AggFill': np.random.randint(10), 'PerturbValue': np.random.random()}
            custom_env = gurobipy.Env()
            for k, v in params.items():
                custom_env.setParam(k, v)
            sth = StandardTestSOCPs.test_socp_0(solver='GUROBI', env=custom_env)
            model = sth.prob.solver_stats.extra_stats
            for k, v in params.items():
                name, p_type, p_val, p_min, p_max, p_def = model.getParamInfo(k)
                self.assertEqual(v, p_val)
        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.GUROBI, TimeLimit=0)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % cp.GUROBI)

    def test_gurobi_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='GUROBI')

    def test_gurobi_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='GUROBI')

    def test_gurobi_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='GUROBI')

    def test_gurobi_lp_3(self) -> None:
        sth = sths.lp_3()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='GUROBI')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)
        StandardTestLPs.test_lp_3(solver='GUROBI', InfUnbdInfo=1)
        StandardTestLPs.test_lp_3(solver='GUROBI', reoptimize=True)

    def test_gurobi_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='GUROBI', reoptimize=True)

    def test_gurobi_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='GUROBI')

    def test_gurobi_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='GUROBI')

    def test_gurobi_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='GUROBI')

    def test_gurobi_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='GUROBI')

    def test_gurobi_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='GUROBI')
        StandardTestSOCPs.test_socp_3ax1(solver='GUROBI')

    def test_gurobi_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='GUROBI')

    def test_gurobi_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='GUROBI')

    def test_gurobi_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='GUROBI')

    def test_gurobi_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='GUROBI')

    def test_gurobi_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='GUROBI')

    def test_gurobi_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='GUROBI', places=2)

    def test_gurobi_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='GUROBI')