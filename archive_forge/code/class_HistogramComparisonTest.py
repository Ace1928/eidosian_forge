import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
class HistogramComparisonTest(ComparisonTestCase):

    def setUp(self):
        """Variations on the constructors in the Elements notebook"""
        np.random.seed(1)
        frequencies1, edges1 = np.histogram([np.random.normal() for i in range(1000)], 20)
        self.hist1 = Histogram((edges1, frequencies1))
        np.random.seed(2)
        frequencies2, edges2 = np.histogram([np.random.normal() for i in range(1000)], 20)
        self.hist2 = Histogram((edges2, frequencies2))
        self.hist3 = Histogram((edges2, frequencies1))
        self.hist4 = Histogram((edges1, frequencies2))

    def test_histograms_equal_1(self):
        self.assertEqual(self.hist1, self.hist1)

    def test_histograms_equal_2(self):
        self.assertEqual(self.hist2, self.hist2)

    def test_histograms_unequal_1(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(self.hist1, self.hist2)

    def test_histograms_unequal_2(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(self.hist1, self.hist3)

    def test_histograms_unequal_3(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(self.hist1, self.hist4)