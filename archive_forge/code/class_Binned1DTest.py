from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
class Binned1DTest(ComparisonTestCase):

    def setUp(self):
        self.values = np.arange(10)
        self.edges = np.arange(11)
        self.dataset1d = Histogram((self.edges, self.values))

    def test_slice_all(self):
        sliced = self.dataset1d[:]
        self.assertEqual(sliced.dimension_values(1), self.values)
        self.assertEqual(sliced.edges, self.edges)

    def test_slice_exclusive_upper(self):
        """Exclusive upper boundary semantics for bin centers"""
        sliced = self.dataset1d[:6.5]
        self.assertEqual(sliced.dimension_values(1), np.arange(6))
        self.assertEqual(sliced.edges, np.arange(7))

    def test_slice_exclusive_upper_exceeded(self):
        """Slightly above the boundary in the previous test"""
        sliced = self.dataset1d[:6.55]
        self.assertEqual(sliced.dimension_values(1), np.arange(7))
        self.assertEqual(sliced.edges, np.arange(8))

    def test_slice_inclusive_lower(self):
        """Inclusive lower boundary semantics for bin centers"""
        sliced = self.dataset1d[3.5:]
        self.assertEqual(sliced.dimension_values(1), np.arange(3, 10))
        self.assertEqual(sliced.edges, np.arange(3, 11))

    def test_slice_inclusive_lower_undershot(self):
        """Inclusive lower boundary semantics for bin centers"""
        sliced = self.dataset1d[3.45:]
        self.assertEqual(sliced.dimension_values(1), np.arange(3, 10))
        self.assertEqual(sliced.edges, np.arange(3, 11))

    def test_slice_bounded(self):
        sliced = self.dataset1d[3.5:6.5]
        self.assertEqual(sliced.dimension_values(1), np.arange(3, 6))
        self.assertEqual(sliced.edges, np.arange(3, 7))

    def test_slice_lower_out_of_bounds(self):
        sliced = self.dataset1d[-3:]
        self.assertEqual(sliced.dimension_values(1), self.values)
        self.assertEqual(sliced.edges, self.edges)

    def test_slice_upper_out_of_bounds(self):
        sliced = self.dataset1d[:12]
        self.assertEqual(sliced.dimension_values(1), self.values)
        self.assertEqual(sliced.edges, self.edges)

    def test_slice_both_out_of_bounds(self):
        sliced = self.dataset1d[-3:13]
        self.assertEqual(sliced.dimension_values(1), self.values)
        self.assertEqual(sliced.edges, self.edges)

    def test_scalar_index(self):
        self.assertEqual(self.dataset1d[4.5], 4)
        self.assertEqual(self.dataset1d[3.7], 3)
        self.assertEqual(self.dataset1d[9.9], 9)

    def test_scalar_index_boundary(self):
        """
        Scalar at boundary indexes next bin.
        (exclusive upper boundary for current bin)
        """
        self.assertEqual(self.dataset1d[4], 4)
        self.assertEqual(self.dataset1d[5], 5)

    def test_scalar_lowest_index(self):
        self.assertEqual(self.dataset1d[0], 0)

    def test_scalar_lowest_index_out_of_bounds(self):
        with self.assertRaises(IndexError):
            self.dataset1d[-1]

    def test_scalar_highest_index_out_of_bounds(self):
        with self.assertRaises(IndexError):
            self.dataset1d[10]

    def test_groupby_kdim(self):
        grouped = self.dataset1d.groupby('x', group_type=Dataset)
        holomap = HoloMap({self.edges[i:i + 2].mean(): Dataset([(i,)], vdims=['Frequency']) for i in range(10)}, kdims=['x'])
        self.assertEqual(grouped, holomap)