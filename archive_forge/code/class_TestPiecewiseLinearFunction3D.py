from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
class TestPiecewiseLinearFunction3D(unittest.TestCase):
    simplices = [[(0, 1), (0, 4), (3, 4)], [(0, 1), (3, 4), (3, 1)], [(3, 4), (3, 7), (0, 7)], [(0, 7), (0, 4), (3, 4)]]

    def make_model(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 3))
        m.x2 = Var(bounds=(1, 7))
        m.g = g
        return m

    def check_pw_linear_approximation(self, m):
        self.assertEqual(len(m.pw._simplices), 4)
        for i, simplex in enumerate(m.pw._simplices):
            for idx in simplex:
                self.assertIn(m.pw._points[idx], self.simplices[i])
        self.assertEqual(len(m.pw._linear_functions), 4)
        assertExpressionsStructurallyEqual(self, m.pw._linear_functions[0](m.x1, m.x2), 3 * m.x1 + 5 * m.x2 - 4, places=7)
        assertExpressionsStructurallyEqual(self, m.pw._linear_functions[1](m.x1, m.x2), 3 * m.x1 + 5 * m.x2 - 4, places=7)
        assertExpressionsStructurallyEqual(self, m.pw._linear_functions[2](m.x1, m.x2), 3 * m.x1 + 11 * m.x2 - 28, places=7)
        assertExpressionsStructurallyEqual(self, m.pw._linear_functions[3](m.x1, m.x2), 3 * m.x1 + 11 * m.x2 - 28, places=7)

    @unittest.skipUnless(scipy_available and numpy_available, 'scipy and/or numpy are not available')
    def test_pw_linear_approx_of_paraboloid_points(self):
        m = self.make_model()
        m.pw = PiecewiseLinearFunction(points=[(0, 1), (0, 4), (0, 7), (3, 1), (3, 4), (3, 7)], function=m.g)
        self.check_pw_linear_approximation(m)

    @unittest.skipUnless(scipy_available, 'scipy is not available')
    def test_pw_linear_approx_tabular_data(self):
        m = self.make_model()
        m.pw = PiecewiseLinearFunction(tabular_data={(0, 1): g(0, 1), (0, 4): g(0, 4), (0, 7): g(0, 7), (3, 1): g(3, 1), (3, 4): g(3, 4), (3, 7): g(3, 7)})
        self.check_pw_linear_approximation(m)

    @unittest.skipUnless(numpy_available, 'numpy are not available')
    def test_pw_linear_approx_of_paraboloid_simplices(self):
        m = self.make_model()
        m.pw = PiecewiseLinearFunction(function=m.g, simplices=self.simplices)
        self.check_pw_linear_approximation(m)

    def test_pw_linear_approx_of_paraboloid_linear_funcs(self):
        m = self.make_model()

        def g1(x1, x2):
            return 3 * x1 + 5 * x2 - 4

        def g2(x1, x2):
            return 3 * x1 + 11 * x2 - 28
        m.pw = PiecewiseLinearFunction(simplices=self.simplices, linear_functions=[g1, g1, g2, g2])
        self.check_pw_linear_approximation(m)

    def test_use_pw_linear_approx_in_constraint(self):
        m = self.make_model()

        def g1(x1, x2):
            return 3 * x1 + 5 * x2 - 4

        def g2(x1, x2):
            return 3 * x1 + 11 * x2 - 28
        m.pw = PiecewiseLinearFunction(simplices=self.simplices, linear_functions=[g1, g1, g2, g2])
        m.c = Constraint(expr=m.pw(m.x1, m.x2) <= 5)
        self.assertEqual(str(m.c.body.expr), 'pw(x1, x2)')
        self.assertIs(m.c.body.expr.pw_linear_function, m.pw)

    @unittest.skipUnless(numpy_available, 'numpy is not available')
    def test_evaluate_pw_linear_function(self):
        m = self.make_model()

        def g1(x1, x2):
            return 3 * x1 + 5 * x2 - 4

        def g2(x1, x2):
            return 3 * x1 + 11 * x2 - 28
        m.pw = PiecewiseLinearFunction(simplices=self.simplices, linear_functions=[g1, g1, g2, g2])
        for x1, x2 in m.pw._points:
            self.assertAlmostEqual(m.pw(x1, x2), m.g(x1, x2))
        self.assertAlmostEqual(m.pw(1, 3), g1(1, 3))
        self.assertAlmostEqual(m.pw(2.5, 6), g2(2.5, 6))
        self.assertAlmostEqual(m.pw(0.2, 4.3), g2(0.2, 4.3))