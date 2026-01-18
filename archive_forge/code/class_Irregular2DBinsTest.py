from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
class Irregular2DBinsTest(ComparisonTestCase):

    def setUp(self):
        lon, lat = np.meshgrid(np.linspace(-20, 20, 6), np.linspace(0, 30, 4))
        lon += lat / 10
        lat += lon / 10
        self.xs = lon
        self.ys = lat
        self.zs = np.arange(24).reshape(4, 6)

    def test_construct_from_dict(self):
        dataset = Dataset((self.xs, self.ys, self.zs), ['x', 'y'], 'z')
        self.assertEqual(dataset.dimension_values('x'), self.xs.T.flatten())
        self.assertEqual(dataset.dimension_values('y'), self.ys.T.flatten())
        self.assertEqual(dataset.dimension_values('z'), self.zs.T.flatten())

    def test_construct_from_xarray(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest('Test requires xarray')
        coords = dict([('lat', (('y', 'x'), self.ys)), ('lon', (('y', 'x'), self.xs))])
        da = xr.DataArray(self.zs, dims=['y', 'x'], coords=coords, name='z')
        dataset = Dataset(da)
        self.assertEqual(dataset.kdims, [Dimension('lat'), Dimension('lon')])
        self.assertEqual(dataset.vdims, [Dimension('z')])
        self.assertEqual(dataset.dimension_values('lon', flat=False), self.xs)
        self.assertEqual(dataset.dimension_values('lat', flat=False), self.ys)
        self.assertEqual(dataset.dimension_values('z'), self.zs.T.flatten())

    def test_construct_3d_from_xarray(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest('Test requires xarray')
        zs = np.arange(48).reshape(2, 4, 6)
        da = xr.DataArray(zs, dims=['z', 'y', 'x'], coords={'lat': (('y', 'x'), self.ys), 'lon': (('y', 'x'), self.xs), 'z': [0, 1]}, name='A')
        dataset = Dataset(da, ['lon', 'lat', 'z'], 'A')
        self.assertEqual(dataset.dimension_values('lon'), self.xs.T.flatten())
        self.assertEqual(dataset.dimension_values('lat'), self.ys.T.flatten())
        self.assertEqual(dataset.dimension_values('z', expanded=False), np.array([0, 1]))
        self.assertEqual(dataset.dimension_values('A'), zs.T.flatten())

    def test_construct_from_xarray_with_invalid_irregular_coordinate_arrays(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest('Test requires xarray')
        zs = np.arange(48 * 6).reshape(2, 4, 6, 6)
        da = xr.DataArray(zs, dims=['z', 'y', 'x', 'b'], coords={'lat': (('y', 'b'), self.ys), 'lon': (('y', 'x'), self.xs), 'z': [0, 1]}, name='A')
        with self.assertRaises(DataError):
            Dataset(da, ['z', 'lon', 'lat'])

    def test_3d_xarray_with_constant_dim_canonicalized_to_2d(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest('Test requires xarray')
        zs = np.arange(24).reshape(1, 4, 6)
        da = xr.DataArray(zs, dims=['z', 'y', 'x'], coords={'lat': (('y', 'x'), self.ys), 'lon': (('y', 'x'), self.xs), 'z': [0]}, name='A')
        dataset = Dataset(da, ['lon', 'lat'], 'A')
        self.assertEqual(dataset.dimension_values('A', flat=False), zs[0])

    def test_groupby_3d_from_xarray(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest('Test requires xarray')
        zs = np.arange(48).reshape(2, 4, 6)
        da = xr.DataArray(zs, dims=['z', 'y', 'x'], coords={'lat': (('y', 'x'), self.ys), 'lon': (('y', 'x'), self.xs), 'z': [0, 1]}, name='A')
        grouped = Dataset(da, ['lon', 'lat', 'z'], 'A').groupby('z')
        hmap = HoloMap({0: Dataset((self.xs, self.ys, zs[0]), ['lon', 'lat'], 'A'), 1: Dataset((self.xs, self.ys, zs[1]), ['lon', 'lat'], 'A')}, kdims='z')
        self.assertEqual(grouped, hmap)

    def test_irregular_transform_replace_kdim(self):
        transformed = Dataset((self.xs, self.ys, self.zs), ['x', 'y'], 'z').transform(x=dim('x') * 2)
        expected = Dataset((self.xs * 2, self.ys, self.zs), ['x', 'y'], 'z')
        self.assertEqual(expected, transformed)

    def test_irregular_transform_replace_vdim(self):
        transformed = Dataset((self.xs, self.ys, self.zs), ['x', 'y'], 'z').transform(z=dim('z') * 2)
        expected = Dataset((self.xs, self.ys, self.zs * 2), ['x', 'y'], 'z')
        self.assertEqual(expected, transformed)