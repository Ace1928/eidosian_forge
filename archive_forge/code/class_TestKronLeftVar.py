from typing import Tuple
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
class TestKronLeftVar(TestKron):
    C_DIMS = [(1, 1), (2, 1), (1, 2), (2, 2)]

    def symvar_kronl(self, param):
        X = cp.Variable(shape=(2, 2), symmetric=True)
        b_val = 1.5 * np.ones((1, 1))
        if param:
            b = cp.Parameter(shape=(1, 1))
            b.value = b_val
        else:
            b = cp.Constant(b_val)
        L = np.array([[0.5, 1], [2, 3]])
        U = np.array([[10, 11], [12, 13]])
        kronX = cp.kron(X, b)
        objective = cp.Minimize(cp.sum(X.flatten()))
        constraints = [U >= kronX, kronX >= L]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.assertItemsAlmostEqual(X.value, np.array([[0.5, 2], [2, 3]]) / 1.5)
        objective = cp.Maximize(cp.sum(X.flatten()))
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.assertItemsAlmostEqual(X.value, np.array([[10, 11], [11, 13]]) / 1.5)
        pass

    def test_symvar_kronl_param(self):
        self.symvar_kronl(param=True)

    def test_symvar_kronl_const(self):
        self.symvar_kronl(param=False)

    def scalar_kronl(self, param):
        y = cp.Variable(shape=(1, 1))
        A_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        L = np.array([[0.5, 1], [2, 3]])
        U = np.array([[10, 11], [12, 13]])
        if param:
            A = cp.Parameter(shape=(2, 2))
            A.value = A_val
        else:
            A = cp.Constant(A_val)
        krony = cp.kron(y, A)
        constraints = [U >= krony, krony >= L]
        objective = cp.Minimize(y)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.assertItemsAlmostEqual(y.value, np.array([[np.max(L / A_val)]]))
        objective = cp.Maximize(y)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.assertItemsAlmostEqual(y.value, np.array([[np.min(U / A_val)]]))
        pass

    def test_scalar_kronl_param(self):
        self.scalar_kronl(param=True)

    def test_scalar_kronl_const(self):
        self.scalar_kronl(param=False)

    def test_gen_kronl_param(self):
        z_dims = (2, 2)
        for c_dims in TestKronLeftVar.C_DIMS:
            Z, C, L, prob = self.make_kron_prob(z_dims, c_dims, param=True, var_left=True, seed=0)
            prob.solve(solver='ECOS', abstol=1e-08, reltol=1e-08)
            self.assertEqual(prob.status, cp.OPTIMAL)
            con_viols = prob.constraints[0].violation()
            self.assertLessEqual(np.max(con_viols), 0.0001)
            self.assertItemsAlmostEqual(Z.value, L, places=4)

    def test_gen_kronr_const(self):
        z_dims = (2, 2)
        for c_dims in TestKronLeftVar.C_DIMS:
            Z, C, L, prob = self.make_kron_prob(z_dims, c_dims, param=False, var_left=True, seed=0)
            prob.solve(solver='ECOS', abstol=1e-08, reltol=1e-08)
            self.assertEqual(prob.status, cp.OPTIMAL)
            con_viols = prob.constraints[0].violation()
            self.assertLessEqual(np.max(con_viols), 0.0001)
            self.assertItemsAlmostEqual(Z.value, L, places=4)