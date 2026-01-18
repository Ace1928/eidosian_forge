import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
class TestProjectedNLP(unittest.TestCase):

    def test_projected(self):
        m = create_pyomo_model()
        nlp = PyomoNLP(m)
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[1]', 'x[2]'])
        expected_names = ['x[0]', 'x[1]', 'x[2]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        self.assertTrue(np.array_equal(projected_nlp.get_primals(), np.asarray([1.0, 2.0, 4.0])))
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(), np.asarray([8.0, 1.0, 9.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)
        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))
        J = 0.0 * J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))
        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))
        H = 0.0 * H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]', 'x[1]'])
        expected_names = ['x[0]', 'x[2]', 'x[1]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        self.assertTrue(np.array_equal(projected_nlp.get_primals(), np.asarray([1.0, 4.0, 2.0])))
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(), np.asarray([8.0, 9.0, 1.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)
        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))
        J = 0.0 * J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))
        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 1.0], [1.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))
        H = 0.0 * H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]', 'y', 'x[1]'])
        expected_names = ['x[0]', 'x[2]', 'y', 'x[1]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        np.testing.assert_equal(projected_nlp.get_primals(), np.asarray([1.0, 4.0, np.nan, 2.0]))
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(), np.asarray([8.0, 9.0, 0.0, 1.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)
        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))
        J = 0.0 * J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))
        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 0.0, 1.0], [1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))
        H = 0.0 * H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]'])
        expected_names = ['x[0]', 'x[2]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        np.testing.assert_equal(projected_nlp.get_primals(), np.asarray([1.0, 4.0]))
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(), np.asarray([8.0, 9.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 4)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 4)
        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 4)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0], [1.0, 1.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))
        J = 0.0 * J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))
        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 4)
        expectedH = np.asarray([[2.0, 1.0], [1.0, 2.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))
        H = 0.0 * H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))