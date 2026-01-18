from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
class TestTriangulationProducesDegenerateSimplices(unittest.TestCase):
    cube_extreme_pt_indices = [{10, 11, 13, 14, 19, 20, 22, 23}, {9, 10, 12, 13, 18, 19, 21, 22}, {0, 1, 3, 4, 9, 10, 12, 13}, {1, 2, 4, 5, 10, 11, 13, 14}, {3, 4, 6, 7, 12, 13, 15, 16}, {4, 5, 7, 8, 13, 14, 16, 17}, {12, 13, 15, 16, 21, 22, 24, 25}, {13, 14, 16, 17, 22, 23, 25, 26}]

    def make_model(self):
        m = ConcreteModel()
        m.f = lambda x1, x2, y: x1 * x2 + y
        m.points = [(-2.0, 0.0, 1.0), (-2.0, 0.0, 4.0), (-2.0, 0.0, 7.0), (-2.0, 1.5, 1.0), (-2.0, 1.5, 4.0), (-2.0, 1.5, 7.0), (-2.0, 3.0, 1.0), (-2.0, 3.0, 4.0), (-2.0, 3.0, 7.0), (-1.5, 0.0, 1.0), (-1.5, 0.0, 4.0), (-1.5, 0.0, 7.0), (-1.5, 1.5, 1.0), (-1.5, 1.5, 4.0), (-1.5, 1.5, 7.0), (-1.5, 3.0, 1.0), (-1.5, 3.0, 4.0), (-1.5, 3.0, 7.0), (-1.0, 0.0, 1.0), (-1.0, 0.0, 4.0), (-1.0, 0.0, 7.0), (-1.0, 1.5, 1.0), (-1.0, 1.5, 4.0), (-1.0, 1.5, 7.0), (-1.0, 3.0, 1.0), (-1.0, 3.0, 4.0), (-1.0, 3.0, 7.0)]
        return m

    @unittest.skipUnless(scipy_available and numpy_available, 'scipy and/or numpy are not available')
    def test_degenerate_simplices_filtered(self):
        m = self.make_model()
        pw = m.approx = PiecewiseLinearFunction(points=m.points, function=m.f)
        self.assertEqual(len(pw._points), 27)
        for p_model, p_pw in zip(m.points, pw._points):
            self.assertEqual(p_model, p_pw)
        self.assertEqual(len(pw._simplices), 48)
        simplex_in_cube = {idx: 0 for idx in range(8)}
        for simplex in pw._simplices:
            for i, vertex_set in enumerate(self.cube_extreme_pt_indices):
                if set(simplex).issubset(vertex_set):
                    simplex_in_cube[i] += 1
            pts = np.array([pw._points[j] for j in simplex]).transpose()
            A = pts[:, 1:] - np.append(pts[:, :2], pts[:, [0]], axis=1)
            self.assertNotEqual(np.linalg.det(A), 0)
        for num in simplex_in_cube.values():
            self.assertEqual(num, 6)

    @unittest.skipUnless(scipy_available and numpy_available, 'scipy and/or numpy are not available')
    def test_redundant_points_logged(self):
        m = self.make_model()
        m.points.append((-2, 0, 1))
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.contrib.piecewise.piecewise_linear_function', level=logging.INFO):
            m.approx = PiecewiseLinearFunction(points=m.points, function=m.f)
        self.assertIn('The Delaunay triangulation dropped the point with index 27 from the triangulation', out.getvalue())

    @unittest.skipUnless(numpy_available, 'numpy is not available')
    def test_user_given_degenerate_simplex_error(self):
        m = self.make_model()
        with self.assertRaisesRegex(ValueError, 'When calculating the hyperplane approximation over the simplex with index 0, the matrix was unexpectedly singular. This likely means that this simplex is degenerate'):
            m.pw = PiecewiseLinearFunction(simplices=[((-2.0, 0.0, 1.0), (-2.0, 0.0, 4.0), (-2.0, 1.5, 1.0), (-2.0, 1.5, 4.0))], function=m.f)

    @unittest.skipUnless(scipy_available and numpy_available, 'scipy and/or numpy are not available')
    def test_simplex_not_numerically_full_rank_but_determinant_nonzero(self):
        m = ConcreteModel()

        def f(x3, x6, x9, x4):
            return -x6 * (0.01 * x4 * x9 + x3) + 0.98 * x3
        points = [(0, 0.85, 1.2, 0), (0.07478, 0.86396, 1.8668, 5), (0, 0.85, 1.8668, 0), (0.07478, 0.86396, 2.18751, 5), (0, 0.86396, 1.2, 0), (0.07478, 0.87971, 2.18751, 5), (0, 0.87971, 1.2, 0), (0.07478, 0.89001, 2.18751, 5), (0.07478, 0.85, 1.2, 0), (0.28333, 0.86396, 2.18751, 5), (0.07478, 0.86396, 1.2, 0), (0.28333, 0.89001, 2.18751, 5), (0.28333, 0.85, 1.2, 0), (0.31332, 0.89001, 2.18751, 5), (0.31332, 0.85, 1.2, 0), (1.2, 0.89001, 2.18751, 5), (0, 0.89001, 1.2, 0), (0.07478, 0.91727, 1.8668, 5), (0, 0.89001, 1.8668, 0), (0.07478, 0.91727, 2.18751, 5), (0, 0.91727, 1.2, 0), (0.07478, 0.93, 2.18751, 5), (0.07478, 0.89001, 1.2, 0), (0.28333, 0.91727, 2.18751, 5), (0.07478, 0.91727, 1.2, 0), (0.28333, 0.93, 2.18751, 5), (0.28333, 0.89001, 1.2, 0), (0.31332, 0.93, 2.18751, 5), (0.31332, 0.89001, 1.2, 0), (1.2, 0.93, 2.18751, 5), (0, 0.85, 2.18751, 0), (0.07478, 0.86396, 3.49134, 5), (0, 0.85, 3.49134, 0), (0.07478, 0.86396, 4, 5), (0, 0.86396, 2.18751, 0), (0.07478, 0.87971, 4, 5), (0, 0.87971, 2.18751, 0), (0.07478, 0.89001, 4, 5), (0.07478, 0.85, 2.18751, 0), (0.28333, 0.86396, 4, 5), (0.07478, 0.86396, 2.18751, 0), (0.28333, 0.89001, 4, 5), (0.28333, 0.85, 2.18751, 0), (0.31332, 0.89001, 4, 5), (0.31332, 0.85, 2.18751, 0), (1.2, 0.89001, 4, 5), (0, 0.89001, 2.18751, 0), (0.07478, 0.91727, 3.49134, 5), (0, 0.89001, 3.49134, 0), (0.07478, 0.91727, 4, 5), (0, 0.91727, 2.18751, 0), (0.07478, 0.93, 3.49134, 5), (0, 0.91727, 3.49134, 0), (0.07478, 0.93, 4, 5), (0.07478, 0.89001, 2.18751, 0), (0.28333, 0.91727, 4, 5), (0.07478, 0.91727, 2.18751, 0), (0.28333, 0.93, 4, 5), (0.28333, 0.89001, 2.18751, 0), (0.31332, 0.93, 4, 5), (0.31332, 0.89001, 2.18751, 0), (1.2, 0.93, 4, 5)]
        m.pw = PiecewiseLinearFunction(points=points, function=f)
        for pt in points:
            self.assertAlmostEqual(m.pw(*pt), f(*pt))