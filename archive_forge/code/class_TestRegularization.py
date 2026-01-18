import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
class TestRegularization(unittest.TestCase):

    def _test_regularization(self, linear_solver):
        m = make_model()
        interface = InteriorPointInterface(m)
        ip_solver = InteriorPointSolver(linear_solver)
        ip_solver.set_interface(interface)
        interface.set_barrier_parameter(0.1)
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        reg_coef = ip_solver.factorize(kkt)
        self.assertAlmostEqual(reg_coef, 0.0001)
        desired_n_neg_evals = ip_solver.interface.n_eq_constraints() + ip_solver.interface.n_ineq_constraints()
        n_pos_evals, n_neg_evals, n_null_evals = linear_solver.get_inertia()
        self.assertEqual(n_null_evals, 0)
        self.assertEqual(n_neg_evals, desired_n_neg_evals)

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_mumps(self):
        solver = MumpsInterface()
        self._test_regularization(solver)

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_scipy(self):
        solver = ScipyInterface(compute_inertia=True)
        self._test_regularization(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is not available')
    def test_ma27(self):
        solver = InteriorPointMA27Interface(icntl_options={1: 0, 2: 0})
        self._test_regularization(solver)

    def _test_regularization_2(self, linear_solver):
        m = make_model_2()
        interface = InteriorPointInterface(m)
        ip_solver = InteriorPointSolver(linear_solver)
        status = ip_solver.solve(interface)
        self.assertEqual(status, InteriorPointStatus.optimal)
        interface.load_primals_into_pyomo_model()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, pyo.exp(-1))

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_mumps_2(self):
        solver = MumpsInterface()
        self._test_regularization_2(solver)

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_scipy_2(self):
        solver = ScipyInterface(compute_inertia=True)
        self._test_regularization_2(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is not available')
    def test_ma27_2(self):
        solver = InteriorPointMA27Interface(icntl_options={1: 0, 2: 0})
        self._test_regularization_2(solver)