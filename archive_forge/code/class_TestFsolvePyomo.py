import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
@unittest.skipUnless(AmplInterface.available(), 'AmplInterface is not available')
class TestFsolvePyomo(unittest.TestCase):

    def test_available_and_version(self):
        solver = pyo.SolverFactory('scipy.fsolve')
        self.assertTrue(solver.available())
        self.assertTrue(solver.license_is_valid())
        sp_version = tuple((int(num) for num in scipy.__version__.split('.')))
        self.assertEqual(sp_version, solver.version())

    def test_solve_simple_nlp(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.fsolve')
        solver.set_options(dict(full_output=False))
        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)

    def test_solve_results_obj(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.fsolve')
        results = solver.solve(m)
        solution = [m.x[1].value, m.x[2].value, m.x[3].value]
        predicted = [0.92846891, -0.22610731, 0.29465397]
        self.assertStructuredAlmostEqual(solution, predicted)
        self.assertEqual(results.problem.number_of_constraints, 3)
        self.assertEqual(results.problem.number_of_variables, 3)
        self.assertEqual(results.solver.termination_condition, pyo.TerminationCondition.feasible)
        msg = 'Solver failed to return an optimal solution'
        with self.assertRaisesRegex(RuntimeError, msg):
            pyo.assert_optimal_termination(results)
        self.assertEqual(results.solver.status, pyo.SolverStatus.ok)

    def test_solve_max_iter(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.fsolve')
        solver.set_options(dict(xtol=1e-09, maxfev=10))
        res = solver.solve(m)
        self.assertNotEqual(res.solver.return_code, 1)
        self.assertIn('has reached maxfev', res.solver.message)

    def test_solve_too_tight_tol(self):
        m, _ = make_simple_model()
        solver = pyo.SolverFactory('scipy.fsolve', options=dict(xtol=0.001, maxfev=20, tol=1e-08))
        msg = 'does not satisfy the function tolerance'
        with self.assertRaisesRegex(RuntimeError, msg):
            res = solver.solve(m)

    def test_with_scalar_model_bad_starting_point(self):
        m, _ = make_scalar_model()
        solver = pyo.SolverFactory('scipy.fsolve')
        res = solver.solve(m)
        predicted_x = 4.90547401
        self.assertNotEqual(predicted_x, m.x.value)

    def test_with_scalar_model_good_starting_point(self):
        m, _ = make_scalar_model()
        m.x.set_value(4.0)
        solver = pyo.SolverFactory('scipy.fsolve')
        res = solver.solve(m)
        predicted_x = 4.90547401
        self.assertAlmostEqual(predicted_x, m.x.value)