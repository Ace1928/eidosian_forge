import datetime as dt
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, XArrayInterface, concat
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import HSV, RGB, Image, ImageStack, QuadMesh
from .test_gridinterface import BaseGridInterfaceTests
from .test_imageinterface import (
class DaskXArrayInterfaceTest(XArrayInterfaceTests):
    """
    Tests for XArray interface wrapping dask arrays
    """

    def setUp(self):
        try:
            import dask.array
        except ImportError:
            raise SkipTest('Dask could not be imported, cannot test dask arrays with XArrayInterface')
        super().setUp()

    def init_column_data(self):
        import dask.array
        self.xs = np.array(range(11))
        self.xs_2 = self.xs ** 2
        self.y_ints = self.xs * 2
        dask_y = dask.array.from_array(np.array(self.y_ints), 2)
        self.dataset_hm = Dataset((self.xs, dask_y), kdims=['x'], vdims=['y'])
        self.dataset_hm_alias = Dataset((self.xs, dask_y), kdims=[('x', 'X')], vdims=[('y', 'Y')])

    def init_grid_data(self):
        import dask.array
        self.grid_xs = [0, 1]
        self.grid_ys = [0.1, 0.2, 0.3]
        self.grid_zs = np.array([[0, 1], [2, 3], [4, 5]])
        dask_zs = dask.array.from_array(self.grid_zs, 2)
        self.dataset_grid = self.element((self.grid_xs, self.grid_ys, dask_zs), kdims=['x', 'y'], vdims=['z'])
        self.dataset_grid_alias = self.element((self.grid_xs, self.grid_ys, dask_zs), kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        self.dataset_grid_inv = self.element((self.grid_xs[::-1], self.grid_ys[::-1], dask_zs), kdims=['x', 'y'], vdims=['z'])

    def test_xarray_dataset_with_scalar_dim_canonicalize(self):
        import dask.array
        xs = [0, 1]
        ys = [0.1, 0.2, 0.3]
        zs = dask.array.from_array(np.array([[[0, 1], [2, 3], [4, 5]]]), 2)
        xrarr = xr.DataArray(zs, coords={'x': xs, 'y': ys, 't': [1]}, dims=['t', 'y', 'x'])
        xrds = xr.Dataset({'v': xrarr})
        ds = Dataset(xrds, kdims=['x', 'y'], vdims=['v'], datatype=['xarray'])
        canonical = ds.dimension_values(2, flat=False)
        self.assertEqual(canonical.ndim, 2)
        expected = np.array([[0, 1], [2, 3], [4, 5]])
        self.assertEqual(canonical, expected)