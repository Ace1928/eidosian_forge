import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
import numpy as np
from scipy.sparse import coo_matrix, tril
from pyomo.contrib import interior_point as ip
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
@unittest.skip('This does not work yet')
class TestWrongNonzeroOrdering(unittest.TestCase):

    def _test_solvers(self, solver, use_tril):
        mat = get_base_matrix(use_tril=use_tril)
        wrong_order_mat = get_base_matrix_wrong_order(use_tril=use_tril)
        stat = solver.do_symbolic_factorization(mat)
        stat = solver.do_numeric_factorization(wrong_order_mat)
        x_true = np.array([1, 2, 3], dtype=np.double)
        rhs = mat * x_true
        x, res = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x, x_true))

    @unittest.skipIf(not scipy_available, 'scipy is needed for interior point scipy tests')
    def test_scipy(self):
        solver = ScipyInterface()
        self._test_solvers(solver, use_tril=False)

    @unittest.skipIf(not mumps_available, 'mumps is needed for interior point mumps tests')
    def test_mumps(self):
        solver = MumpsInterface()
        self._test_solvers(solver, use_tril=True)

    @unittest.skipIf(not ma27_available, 'MA27 is needed for interior point MA27 tests')
    def test_ma27(self):
        solver = InteriorPointMA27Interface()
        self._test_solvers(solver, use_tril=True)