import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
@unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
class TestCyIpoptSolver(unittest.TestCase):

    def test_model1(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(CyIpoptNLP(nlp))
        x, info = solver.solve(tee=False)
        x_sol = np.array([3.85958688, 4.67936007, 3.10358931])
        y_sol = np.array([-1.0, 53.90357665])
        self.assertTrue(np.allclose(x, x_sol, rtol=0.0001))
        nlp.set_primals(x)
        nlp.set_duals(y_sol)
        self.assertAlmostEqual(nlp.evaluate_objective(), -428.6362455416348, places=5)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=0.0001))

    def test_model1_with_scaling(self):
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.o] = 1e-06
        m.scaling_factor[m.c] = 2.0
        m.scaling_factor[m.d] = 3.0
        m.scaling_factor[m.x[1]] = 4.0
        cynlp = CyIpoptNLP(PyomoNLP(m))
        options = {'nlp_scaling_method': 'user-scaling', 'output_file': '_cyipopt-scaling.log', 'file_print_level': 10, 'max_iter': 0}
        solver = CyIpoptSolver(cynlp, options=options)
        x, info = solver.solve()
        with open('_cyipopt-scaling.log', 'r') as fd:
            solver_trace = fd.read()
        cynlp.close()
        os.remove('_cyipopt-scaling.log')
        self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
        self.assertIn('output_file = _cyipopt-scaling.log', solver_trace)
        self.assertIn('objective scaling factor = 1e-06', solver_trace)
        self.assertIn('x scaling provided', solver_trace)
        self.assertIn('c scaling provided', solver_trace)
        self.assertIn('d scaling provided', solver_trace)
        self.assertIn('DenseVector "x scaling vector" with 3 elements:', solver_trace)
        self.assertIn('x scaling vector[    1]= 1.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    2]= 1.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    3]= 4.0000000000000000e+00', solver_trace)
        self.assertIn('DenseVector "c scaling vector" with 1 elements:', solver_trace)
        self.assertIn('c scaling vector[    1]= 2.0000000000000000e+00', solver_trace)
        self.assertIn('DenseVector "d scaling vector" with 1 elements:', solver_trace)
        self.assertIn('d scaling vector[    1]= 3.0000000000000000e+00', solver_trace)

    def test_model2(self):
        model = create_model2()
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(CyIpoptNLP(nlp))
        x, info = solver.solve(tee=False)
        x_sol = np.array([3.0, 1.99997807])
        y_sol = np.array([0.00017543])
        self.assertTrue(np.allclose(x, x_sol, rtol=0.0001))
        nlp.set_primals(x)
        nlp.set_duals(y_sol)
        self.assertAlmostEqual(nlp.evaluate_objective(), -31.000000057167462, places=5)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=0.0001))

    def test_model3(self):
        G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = np.array([[1, 0, 1], [0, 1, 1]])
        b = np.array([3, 0])
        c = np.array([-8, -3, -3])
        model = create_model3(G, A, b, c)
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(CyIpoptNLP(nlp))
        x, info = solver.solve(tee=False)
        x_sol = np.array([2.0, -1.0, 1.0])
        y_sol = np.array([-3.0, 2.0])
        self.assertTrue(np.allclose(x, x_sol, rtol=0.0001))
        nlp.set_primals(x)
        nlp.set_duals(y_sol)
        self.assertAlmostEqual(nlp.evaluate_objective(), -3.5, places=5)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=0.0001))

    def test_options(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(CyIpoptNLP(nlp), options={'max_iter': 1})
        x, info = solver.solve(tee=False)
        nlp.set_primals(x)
        self.assertAlmostEqual(nlp.evaluate_objective(), -508.79028, places=5)

    @unittest.skipUnless(cyipopt_available and cyipopt_ge_1_3, 'cyipopt version < 1.3.0')
    def test_hs071_evalerror(self):
        m = make_hs071_model()
        solver = pyo.SolverFactory('cyipopt')
        res = solver.solve(m, tee=True)
        x = list(m.x[:].value)
        expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
        np.testing.assert_allclose(x, expected_x)

    def test_hs071_evalerror_halt(self):
        m = make_hs071_model()
        solver = pyo.SolverFactory('cyipopt', halt_on_evaluation_error=True)
        msg = 'Error in AMPL evaluation'
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            res = solver.solve(m, tee=True)

    @unittest.skipIf(not cyipopt_available or cyipopt_ge_1_3, 'cyipopt version >= 1.3.0')
    def test_hs071_evalerror_old_cyipopt(self):
        m = make_hs071_model()
        solver = pyo.SolverFactory('cyipopt')
        msg = 'Error in AMPL evaluation'
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            res = solver.solve(m, tee=True)