import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
class DimensionValuesTest(ComparisonTestCase):

    def setUp(self):
        self.values1 = [0, 1, 2, 3, 4, 5, 6]
        self.values2 = ['a', 'b', 'c', 'd']
        self.duplicates1 = [0, 1, 0, 2, 3, 4, 3, 2, 5, 5, 6]
        self.duplicates2 = ['a', 'b', 'b', 'a', 'c', 'a', 'c', 'd', 'd']

    def test_dimension_values_list1(self):
        dim = Dimension('test', values=self.values1)
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_list2(self):
        dim = Dimension('test', values=self.values2)
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_list_duplicates1(self):
        dim = Dimension('test', values=self.duplicates1)
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_list_duplicates2(self):
        dim = Dimension('test', values=self.duplicates2)
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_array1(self):
        dim = Dimension('test', values=np.array(self.values1))
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_array2(self):
        dim = Dimension('test', values=np.array(self.values2))
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_array_duplicates1(self):
        dim = Dimension('test', values=np.array(self.duplicates1))
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_array_duplicates2(self):
        dim = Dimension('test', values=np.array(self.duplicates2))
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_series1(self):
        df = pd.DataFrame({'col': self.values1})
        dim = Dimension('test', values=df['col'])
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_series2(self):
        df = pd.DataFrame({'col': self.values2})
        dim = Dimension('test', values=df['col'])
        self.assertEqual(dim.values, self.values2)

    def test_dimension_values_series_duplicates1(self):
        df = pd.DataFrame({'col': self.duplicates1})
        dim = Dimension('test', values=df['col'])
        self.assertEqual(dim.values, self.values1)

    def test_dimension_values_series_duplicates2(self):
        df = pd.DataFrame({'col': self.duplicates2})
        dim = Dimension('test', values=df['col'])
        self.assertEqual(dim.values, self.values2)