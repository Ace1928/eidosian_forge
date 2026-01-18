from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
class Binned2DTest(ComparisonTestCase):

    def setUp(self):
        n = 4
        self.xs = np.logspace(1, 3, n)
        self.ys = np.linspace(1, 10, n)
        self.zs = np.arange((n - 1) ** 2).reshape(n - 1, n - 1)
        self.dataset2d = QuadMesh((self.xs, self.ys, self.zs))

    def test_qmesh_index_lower_left(self):
        self.assertEqual(self.dataset2d[10, 1], 0)

    def test_qmesh_index_lower_right(self):
        self.assertEqual(self.dataset2d[800, 3.9], 2)

    def test_qmesh_index_top_left(self):
        self.assertEqual(self.dataset2d[10, 9.9], 6)

    def test_qmesh_index_top_right(self):
        self.assertEqual(self.dataset2d[216, 7], 8)

    def test_qmesh_index_xcoords(self):
        sliced = QuadMesh((self.xs[2:4], self.ys, self.zs[:, 2:3]))
        self.assertEqual(self.dataset2d[300, :], sliced)

    def test_qmesh_index_ycoords(self):
        sliced = QuadMesh((self.xs, self.ys[-2:], self.zs[-1:, :]))
        self.assertEqual(self.dataset2d[:, 7], sliced)

    def test_qmesh_slice_xcoords(self):
        sliced = QuadMesh((self.xs[1:], self.ys, self.zs[:, 1:]))
        self.assertEqual(self.dataset2d[100:1000, :], sliced)

    def test_qmesh_slice_ycoords(self):
        sliced = QuadMesh((self.xs, self.ys[:-1], self.zs[:-1, :]))
        self.assertEqual(self.dataset2d[:, 2:7], sliced)

    def test_qmesh_slice_xcoords_ycoords(self):
        sliced = QuadMesh((self.xs[1:], self.ys[:-1], self.zs[:-1, 1:]))
        self.assertEqual(self.dataset2d[100:1000, 2:7], sliced)

    def test_groupby_xdim(self):
        grouped = self.dataset2d.groupby('x', group_type=Dataset)
        holomap = HoloMap({(self.xs[i] + np.diff(self.xs[i:i + 2]) / 2.0)[0]: Dataset((self.ys, self.zs[:, i]), 'y', 'z') for i in range(3)}, kdims=['x'])
        self.assertEqual(grouped, holomap)

    def test_groupby_ydim(self):
        grouped = self.dataset2d.groupby('y', group_type=Dataset)
        holomap = HoloMap({self.ys[i:i + 2].mean(): Dataset((self.xs, self.zs[i]), 'x', 'z') for i in range(3)}, kdims=['y'])
        self.assertEqual(grouped, holomap)

    def test_qmesh_transform_replace_kdim(self):
        transformed = self.dataset2d.transform(x=dim('x') * 2)
        expected = QuadMesh((self.xs * 2, self.ys, self.zs))
        self.assertEqual(expected, transformed)

    def test_qmesh_transform_replace_vdim(self):
        transformed = self.dataset2d.transform(z=dim('z') * 2)
        expected = QuadMesh((self.xs, self.ys, self.zs * 2))
        self.assertEqual(expected, transformed)