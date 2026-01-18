import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
@unittest.skipUnless(AmplInterface.available(), 'AmplInterface is not available')
class TestRootPyomo(unittest.TestCase):

    def test_available_and_version(self):
        solver = pyo.SolverFactory('scipy.root')
        self.assertTrue(solver.available())
        self.assertTrue(solver.license_is_valid())
        sp_version = tuple((int(num) for num in scipy.__version__.split('.')))
        self.assertEqual(sp_version, solver.version())

    def test_solve_simple_nlp(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.root')
        solver.set_options(dict(tol=1e-07))
        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

    def test_solve_simple_nlp_levenberg_marquardt(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.root')
        solver.set_options(dict(tol=1e-07, method='lm'))
        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

    def test_bad_method(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.root')
        solver.set_options(dict(tol=1e-07, method='some-solver'))
        with self.assertRaisesRegex(ValueError, 'not in domain'):
            results = solver.solve(m)

    def test_solver_results_obj(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.root')
        solver.set_options(dict(tol=1e-07))
        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)
        self.assertEqual(results.problem.number_of_constraints, 3)
        self.assertEqual(results.problem.number_of_variables, 3)
        self.assertEqual(results.solver.return_code, 1)
        self.assertEqual(results.solver.termination_condition, pyo.TerminationCondition.feasible)
        self.assertEqual(results.solver.message, 'The solution converged.')

    def test_solver_results_obj_levenberg_marquardt(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.root')
        solver.set_options(dict(tol=1e-07, method='lm'))
        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)
        self.assertEqual(results.problem.number_of_constraints, 3)
        self.assertEqual(results.problem.number_of_variables, 3)
        self.assertEqual(results.solver.termination_condition, pyo.TerminationCondition.feasible)
        self.assertIn('The relative error between two consecutive iterates', results.solver.message)