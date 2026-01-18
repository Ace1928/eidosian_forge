import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import (
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models
@unittest.skipUnless(SolverFactory('ipopt').available(), 'Ipopt not available')
class TestGDPoptEnumerate_ipopt_tests(unittest.TestCase):

    def test_infeasible_GDP(self):
        m = models.make_infeasible_gdp_model()
        results = SolverFactory('gdpopt.enumerate').solve(m)
        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)
        self.assertEqual(results.problem.lower_bound, float('inf'))

    def test_algorithm_specified_to_solve(self):
        m = models.twoDisj_twoCircles_easy()
        results = SolverFactory('gdpopt').solve(m, algorithm='enumerate', tee=True)
        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(results.problem.lower_bound, 9)
        self.assertAlmostEqual(results.problem.upper_bound, 9)
        self.assertAlmostEqual(value(m.x), 2)
        self.assertAlmostEqual(value(m.y), 7)
        self.assertTrue(value(m.upper_circle.indicator_var))
        self.assertFalse(value(m.lower_circle.indicator_var))