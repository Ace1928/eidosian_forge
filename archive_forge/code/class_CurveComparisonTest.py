import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
class CurveComparisonTest(ComparisonTestCase):

    def setUp(self):
        """Variations on the constructors in the Elements notebook"""
        self.curve1 = Curve([(0.1 * i, np.sin(0.1 * i)) for i in range(100)])
        self.curve2 = Curve([(0.1 * i, np.sin(0.1 * i)) for i in range(101)])

    def test_curves_equal(self):
        self.assertEqual(self.curve1, self.curve1)

    def test_curves_unequal(self):
        try:
            self.assertEqual(self.curve1, self.curve2)
        except AssertionError as e:
            if not str(e).startswith('Curve not of matching length, 100 vs. 101'):
                raise self.failureException('Curve mismatch error not raised.')