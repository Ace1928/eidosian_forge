import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
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