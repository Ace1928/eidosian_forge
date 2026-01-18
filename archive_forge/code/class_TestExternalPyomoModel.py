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
class TestExternalPyomoModel(unittest.TestCase):

    def test_evaluate_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            resid = external_model.evaluate_equality_constraints()
            self.assertAlmostEqual(resid[0], model.evaluate_residual(x[0]), delta=1e-08)

    def test_jacobian_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_equality_constraints()
            self.assertAlmostEqual(jac.toarray()[0][0], model.evaluate_jacobian(x[0]), delta=1e-08)

    def test_hessian_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessians_of_residuals()
            self.assertAlmostEqual(hess[0][0, 0], model.evaluate_hessian(x[0]), delta=1e-08)

    def test_external_hessian_SimpleModel1(self):
        model = SimpleModel1()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x[0])
            self.assertAlmostEqual(hess[0][0, 0], expected_hess, delta=1e-08)

    def test_evaluate_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            resid = external_model.evaluate_equality_constraints()
            self.assertAlmostEqual(resid[0], model.evaluate_residual(x[0]), delta=1e-08)

    def test_jacobian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_equality_constraints()
            self.assertAlmostEqual(jac.toarray()[0][0], model.evaluate_jacobian(x[0]), delta=1e-07)

    def test_hessian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessians_of_residuals()
            self.assertAlmostEqual(hess[0][0, 0], model.evaluate_hessian(x[0]), delta=1e-07)

    def test_external_jacobian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_external_variables()
            expected_jac = model.evaluate_external_jacobian(x[0])
            self.assertAlmostEqual(jac[0, 0], expected_jac, delta=1e-08)

    def test_external_hessian_SimpleModel2(self):
        model = SimpleModel2()
        m = model.make_model()
        x_init_list = [[-5.0], [-4.0], [-3.0], [-1.5], [0.5], [1.0], [2.0], [3.5]]
        external_model = ExternalPyomoModel([m.x], [m.y], [m.residual_eqn], [m.external_eqn])
        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x[0])
            self.assertAlmostEqual(hess[0][0, 0], expected_hess, delta=1e-07)

    def test_external_jacobian_Model2by2(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_external_variables()
            expected_jac = model.evaluate_external_jacobian(x)
            np.testing.assert_allclose(jac, expected_jac, rtol=1e-08)

    def test_external_hessian_Model2by2(self):
        model = Model2by2()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [0.5, 1.0, 1.5, 2.5, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x)
            for matrix1, matrix2 in zip(hess, expected_hess):
                np.testing.assert_allclose(matrix1, matrix2, rtol=1e-08)

    def test_external_jacobian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_external_variables()
            expected_jac = model.evaluate_external_jacobian(x)
            np.testing.assert_allclose(jac, expected_jac, rtol=1e-08)

    def test_external_hessian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessian_external_variables()
            expected_hess = model.evaluate_external_hessian(x)
            for matrix1, matrix2 in zip(hess, expected_hess):
                np.testing.assert_allclose(matrix1, matrix2, rtol=1e-08)

    def test_evaluate_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
        for x in x_init_list:
            external_model.set_input_values(x)
            resid = external_model.evaluate_equality_constraints()
            expected_resid = model.evaluate_residual(x)
            np.testing.assert_allclose(resid, expected_resid, rtol=1e-08)

    def test_jacobian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
        for x in x_init_list:
            external_model.set_input_values(x)
            jac = external_model.evaluate_jacobian_equality_constraints()
            expected_jac = model.evaluate_jacobian(x)
            np.testing.assert_allclose(jac.toarray(), expected_jac, rtol=1e-08, atol=1e-08)

    def test_hessian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
        for x in x_init_list:
            external_model.set_input_values(x)
            hess = external_model.evaluate_hessians_of_residuals()
            expected_hess = model.evaluate_hessian(x)
            np.testing.assert_allclose(hess, expected_hess, rtol=1e-08)

    def test_evaluate_hessian_lagrangian_SimpleModel2x2_1(self):
        model = SimpleModel2by2_1()
        m = model.make_model()
        m.x[0].set_value(1.0)
        m.x[1].set_value(2.0)
        m.y[0].set_value(3.0)
        m.y[1].set_value(4.0)
        x0_init_list = [-5.0, -3.0, 0.5, 1.0, 2.5]
        x1_init_list = [-4.5, -2.3, 0.0, 1.0, 4.1]
        x_init_list = list(itertools.product(x0_init_list, x1_init_list))
        external_model = ExternalPyomoModel(list(m.x.values()), list(m.y.values()), list(m.residual_eqn.values()), list(m.external_eqn.values()))
        for x in x_init_list:
            external_model.set_input_values(x)
            external_model.set_equality_constraint_multipliers([1.0, 1.0])
            hess_lag = external_model.evaluate_hessian_equality_constraints()
            hess_lag = hess_lag.toarray()
            expected_hess = model.evaluate_hessian(x)
            expected_hess_lag = np.tril(expected_hess[0] + expected_hess[1])
            np.testing.assert_allclose(hess_lag, expected_hess_lag, rtol=1e-08)