import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
class DimensionEqualityTest(ComparisonTestCase):

    def test_simple_dim_equality(self):
        dim1 = Dimension('test')
        dim2 = Dimension('test')
        self.assertEqual(dim1 == dim2, True)

    def test_simple_str_equality(self):
        dim1 = Dimension('test')
        dim2 = Dimension('test')
        self.assertEqual(dim1 == str(dim2), True)

    def test_simple_dim_inequality(self):
        dim1 = Dimension('test1')
        dim2 = Dimension('test2')
        self.assertEqual(dim1 == dim2, False)

    def test_simple_str_inequality(self):
        dim1 = Dimension('test1')
        dim2 = Dimension('test2')
        self.assertEqual(dim1 == str(dim2), False)

    def test_label_dim_inequality(self):
        dim1 = Dimension(('test', 'label1'))
        dim2 = Dimension(('test', 'label2'))
        self.assertEqual(dim1 == dim2, False)

    def test_label_str_equality(self):
        dim1 = Dimension(('test', 'label1'))
        dim2 = Dimension(('test', 'label2'))
        self.assertEqual(dim1 == str(dim2), True)

    def test_weak_dim_equality(self):
        dim1 = Dimension('test', cyclic=True, unit='m', type=float)
        dim2 = Dimension('test', cyclic=False, unit='km', type=int)
        self.assertEqual(dim1 == dim2, True)

    def test_weak_str_equality(self):
        dim1 = Dimension('test', cyclic=True, unit='m', type=float)
        dim2 = Dimension('test', cyclic=False, unit='km', type=int)
        self.assertEqual(dim1 == str(dim2), True)