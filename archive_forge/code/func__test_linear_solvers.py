import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
import numpy as np
from scipy.sparse import coo_matrix, tril
from pyomo.contrib import interior_point as ip
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
def _test_linear_solvers(self, solver):
    mat = get_base_matrix(use_tril=False)
    zero_mat = mat.copy()
    zero_mat.data.fill(0)
    stat = solver.do_symbolic_factorization(zero_mat)
    self.assertEqual(stat.status, LinearSolverStatus.successful)
    stat = solver.do_numeric_factorization(mat)
    self.assertEqual(stat.status, LinearSolverStatus.successful)
    x_true = np.array([1, 2, 3], dtype=np.double)
    rhs = mat * x_true
    x, res = solver.do_back_solve(rhs)
    self.assertTrue(np.allclose(x, x_true))
    x_true = np.array([4, 2, 3], dtype=np.double)
    rhs = mat * x_true
    x, res = solver.do_back_solve(rhs)
    self.assertTrue(np.allclose(x, x_true))