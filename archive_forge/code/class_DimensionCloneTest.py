import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
class DimensionCloneTest(ComparisonTestCase):

    def test_simple_clone(self):
        dim = Dimension('test')
        self.assertEqual(dim.name, 'test')
        self.assertEqual(dim.clone('bar').name, 'bar')

    def test_simple_label_clone(self):
        dim = Dimension('test')
        self.assertEqual(dim.name, 'test')
        clone = dim.clone(label='label')
        self.assertEqual(clone.name, 'test')
        self.assertEqual(clone.label, 'label')

    def test_simple_values_clone(self):
        dim = Dimension('test', values=[1, 2, 3])
        self.assertEqual(dim.values, [1, 2, 3])
        clone = dim.clone(values=[4, 5, 6])
        self.assertEqual(clone.name, 'test')
        self.assertEqual(clone.values, [4, 5, 6])

    def test_tuple_clone(self):
        dim = Dimension('test')
        self.assertEqual(dim.name, 'test')
        clone = dim.clone(('test', 'A test'))
        self.assertEqual(clone.name, 'test')
        self.assertEqual(clone.label, 'A test')