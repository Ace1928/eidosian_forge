import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import cyipopt_available
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
class TestCyIpoptNLP(unittest.TestCase):

    def test_model1_CyIpoptNLP(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        cynlp = CyIpoptNLP(nlp)
        self._check_model1(nlp, cynlp)

    def test_model1_CyIpoptNLP_scaling(self):
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.o] = 1e-06
        m.scaling_factor[m.c] = 2.0
        m.scaling_factor[m.d] = 3.0
        m.scaling_factor[m.x[1]] = 4.0
        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling == 1e-06)
        self.assertTrue(len(x_scaling) == 3)
        self.assertTrue(x_scaling[0] == 1.0)
        self.assertTrue(x_scaling[1] == 1.0)
        self.assertTrue(x_scaling[2] == 4.0)
        self.assertTrue(len(g_scaling) == 2)
        self.assertTrue(g_scaling[0] == 3.0)
        self.assertTrue(g_scaling[1] == 2.0)
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.c] = 2.0
        m.scaling_factor[m.d] = 3.0
        m.scaling_factor[m.x[1]] = 4.0
        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling == 1.0)
        self.assertTrue(len(x_scaling) == 3)
        self.assertTrue(x_scaling[0] == 1.0)
        self.assertTrue(x_scaling[1] == 1.0)
        self.assertTrue(x_scaling[2] == 4.0)
        self.assertTrue(len(g_scaling) == 2)
        self.assertTrue(g_scaling[0] == 3.0)
        self.assertTrue(g_scaling[1] == 2.0)
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.o] = 1e-06
        m.scaling_factor[m.c] = 2.0
        m.scaling_factor[m.d] = 3.0
        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling == 1e-06)
        self.assertTrue(len(x_scaling) == 3)
        self.assertTrue(x_scaling[0] == 1.0)
        self.assertTrue(x_scaling[1] == 1.0)
        self.assertTrue(x_scaling[2] == 1.0)
        self.assertTrue(len(g_scaling) == 2)
        self.assertTrue(g_scaling[0] == 3.0)
        self.assertTrue(g_scaling[1] == 2.0)
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.o] = 1e-06
        m.scaling_factor[m.d] = 3.0
        m.scaling_factor[m.x[1]] = 4.0
        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling == 1e-06)
        self.assertTrue(len(x_scaling) == 3)
        self.assertTrue(x_scaling[0] == 1.0)
        self.assertTrue(x_scaling[1] == 1.0)
        self.assertTrue(x_scaling[2] == 4.0)
        self.assertTrue(len(g_scaling) == 2)
        self.assertTrue(g_scaling[0] == 3.0)
        self.assertTrue(g_scaling[1] == 1.0)
        m = create_model1()
        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling is None)
        self.assertTrue(x_scaling is None)
        self.assertTrue(g_scaling is None)

    def _check_model1(self, nlp, cynlp):
        expected_xinit = np.asarray([4.0, 4.0, 4.0], dtype=np.float64)
        xinit = cynlp.x_init()
        self.assertTrue(np.array_equal(xinit, expected_xinit))
        expected_xlb = list()
        for v in nlp.get_pyomo_variables():
            if v.lb == None:
                expected_xlb.append(-np.inf)
            else:
                expected_xlb.append(v.lb)
        expected_xlb = np.asarray(expected_xlb)
        xlb = cynlp.x_lb()
        self.assertTrue(np.array_equal(xlb, expected_xlb))
        expected_xub = list()
        for v in nlp.get_pyomo_variables():
            if v.ub == None:
                expected_xub.append(np.inf)
            else:
                expected_xub.append(v.ub)
        expected_xub = np.asarray(expected_xub)
        xub = cynlp.x_ub()
        self.assertTrue(np.array_equal(xub, expected_xub))
        expected_glb = np.asarray([-np.inf, 0.0], dtype=np.float64)
        glb = cynlp.g_lb()
        self.assertTrue(np.array_equal(glb, expected_glb))
        expected_gub = np.asarray([18, 0.0], dtype=np.float64)
        gub = cynlp.g_ub()
        print(expected_gub)
        print(gub)
        self.assertTrue(np.array_equal(gub, expected_gub))
        x = cynlp.x_init()
        self.assertEqual(cynlp.objective(x), -504)
        expected = np.asarray([-576, 8, 64], dtype=np.float64)
        self.assertTrue(np.allclose(expected, cynlp.gradient(x)))
        expected = np.asarray([20, -5], dtype=np.float64)
        constraints = cynlp.constraints(x)
        self.assertTrue(np.allclose(expected, constraints))
        expected = np.asarray([[8.0, 0, 1.0], [0.0, 8.0, 1.0]])
        spexpected = spa.coo_matrix(expected).todense()
        rows, cols = cynlp.jacobianstructure()
        values = cynlp.jacobian(x)
        jac = spa.coo_matrix((values, (rows, cols)), shape=(len(constraints), len(x))).todense()
        self.assertTrue(np.allclose(spexpected, jac))
        y = constraints.copy()
        y.fill(1.0)
        rows, cols = cynlp.hessianstructure()
        values = cynlp.hessian(x, y, obj_factor=1.0)
        hess_lower = spa.coo_matrix((values, (rows, cols)), shape=(len(x), len(x))).todense()
        expected_hess_lower = np.asarray([[-286.0, 0.0, 0.0], [0.0, 4.0, 0.0], [-144.0, 0.0, 192.0]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_hess_lower, hess_lower))