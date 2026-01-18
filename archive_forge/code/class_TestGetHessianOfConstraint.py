import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
class TestGetHessianOfConstraint(unittest.TestCase):

    def test_simple_model_1(self):
        model = SimpleModel1()
        m = model.make_model()
        m.x.set_value(2.0)
        m.y.set_value(2.0)
        con = m.residual_eqn
        expected_hess = np.array([[2.0, 0.0], [0.0, 2.0]])
        hess = get_hessian_of_constraint(con)
        self.assertTrue(np.all(expected_hess == hess.toarray()))
        expected_hess = np.array([[2.0]])
        hess = get_hessian_of_constraint(con, [m.x])
        self.assertTrue(np.all(expected_hess == hess.toarray()))
        con = m.external_eqn
        expected_hess = np.array([[0.0, 1.0], [1.0, 0.0]])
        hess = get_hessian_of_constraint(con)
        self.assertTrue(np.all(expected_hess == hess.toarray()))

    def test_polynomial(self):
        m = pyo.ConcreteModel()
        n_x = 3
        x1 = 1.1
        x2 = 1.2
        x3 = 1.3
        m.x = pyo.Var(range(1, n_x + 1), initialize={1: x1, 2: x2, 3: x3})
        m.eqn = pyo.Constraint(expr=5 * m.x[1] ** 5 + 5 * m.x[1] ** 4 * m.x[2] + 5 * m.x[1] ** 3 * m.x[2] * m.x[3] + 5 * m.x[1] * m.x[2] ** 2 * m.x[3] ** 2 + 4 * m.x[1] ** 2 * m.x[2] * m.x[3] + 4 * m.x[2] ** 2 * m.x[3] ** 2 + 4 * m.x[3] ** 4 + 3 * m.x[1] * m.x[2] * m.x[3] + 3 * m.x[2] ** 3 + 3 * m.x[2] ** 2 * m.x[3] + 2 * m.x[1] * m.x[2] + 2 * m.x[2] * m.x[3] == 0)
        rcd = []
        rcd.append((0, 0, 5 * 5 * 4 * x1 ** 3 + 5 * 4 * 3 * x1 ** 2 * x2 + 5 * 3 * 2 * x1 * x2 * x3 + 4 * 2 * 1 * x2 * x3))
        rcd.append((1, 1, 5 * x1 * 2 * x3 ** 2 + 4 * 2 * x3 ** 2 + 3 * 3 * 2 * x2 + 3 * 2 * x3))
        rcd.append((2, 2, 5 * x1 * x2 ** 2 * 2 + 4 * x2 ** 2 * 2 + 4 * 4 * 3 * x3 ** 2))
        rcd.append((1, 0, 5 * 4 * x1 ** 3 + 5 * 3 * x1 ** 2 * x3 + 5 * 2 * x2 * x3 ** 2 + 4 * 2 * x1 * x3 + 3 * x3 + 2))
        rcd.append((2, 0, 5 * 3 * x1 ** 2 * x2 + 5 * x2 ** 2 * 2 * x3 + 4 * 2 * x1 * x2 + 3 * x2))
        rcd.append((2, 1, 5 * x1 ** 3 + 5 * x1 * 2 * x2 * 2 * x3 + 4 * x1 ** 2 + 4 * 2 * x2 * 2 * x3 + 3 * x1 + 3 * 2 * x2 + 2))
        row = [r for r, _, _ in rcd]
        col = [c for _, c, _ in rcd]
        data = [d for _, _, d in rcd]
        expected_hess = sps.coo_matrix((data, (row, col)), shape=(n_x, n_x))
        expected_hess_array = expected_hess.toarray()
        expected_hess_array = expected_hess_array + np.transpose(expected_hess_array) - np.diag(np.diagonal(expected_hess_array))
        hess = get_hessian_of_constraint(m.eqn, list(m.x.values()))
        hess_array = hess.toarray()
        np.testing.assert_allclose(expected_hess_array, hess_array, rtol=1e-08)

    def test_unused_variable(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=1.0)
        m.z = pyo.Var(initialize=1.0)
        m.eqn = pyo.Constraint(expr=m.x ** 2 + m.y ** 2 == 1.0)
        variables = [m.x, m.y, m.z]
        expected_hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
        hess = get_hessian_of_constraint(m.eqn, variables).toarray()
        np.testing.assert_allclose(hess, expected_hess, rtol=1e-08)

    def test_explicit_zeros(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=0.0)
        m.eqn = pyo.Constraint(expr=m.x ** 2 + m.y ** 3 == 1.0)
        variables = [m.x, m.y]
        row = np.array([0, 1])
        col = np.array([0, 1])
        data = np.array([2.0, 0.0])
        expected_hess = sps.coo_matrix((data, (row, col)), shape=(2, 2))
        hess = get_hessian_of_constraint(m.eqn, variables)
        np.testing.assert_allclose(hess.row, row, atol=0)
        np.testing.assert_allclose(hess.col, col, atol=0)
        np.testing.assert_allclose(hess.data, data, rtol=1e-08)