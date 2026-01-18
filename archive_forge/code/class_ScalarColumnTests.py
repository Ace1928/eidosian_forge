import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
class ScalarColumnTests:
    """
    Tests for interfaces that allow on or more columns to be of scalar
    types.
    """
    __test__ = False

    def test_dataset_scalar_constructor(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.dimension_values('A'), np.ones(10))

    def test_dataset_scalar_length(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(len(ds), 10)

    def test_dataset_scalar_array(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.array(), np.column_stack([np.ones(10), np.arange(10)]))

    def test_dataset_scalar_select(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.select(A=1).dimension_values('B'), np.arange(10))

    def test_dataset_scalar_select_expr(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.select(selection_expr=dim('A') == 1).dimension_values('B'), np.arange(10))

    def test_dataset_scalar_empty_select(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.select(A=0).dimension_values('B'), np.array([]))

    def test_dataset_scalar_empty_select_expr(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.select(selection_expr=dim('A') == 0).dimension_values('B'), np.array([]))

    def test_dataset_scalar_sample(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.sample([(1,)]).dimension_values('B'), np.arange(10))

    def test_dataset_scalar_sort(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)[::-1]}, kdims=['A', 'B'])
        self.assertEqual(ds.sort().dimension_values('B'), np.arange(10))

    def test_dataset_scalar_groupby(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        groups = ds.groupby('A')
        self.assertEqual(groups, HoloMap({1: Dataset({'B': np.arange(10)}, 'B')}, 'A'))

    def test_dataset_scalar_iloc(self):
        ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
        self.assertEqual(ds.iloc[:5], Dataset({'A': 1, 'B': np.arange(5)}, kdims=['A', 'B']))