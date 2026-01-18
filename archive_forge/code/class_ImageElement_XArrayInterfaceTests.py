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
class ImageElement_XArrayInterfaceTests(BaseImageElementInterfaceTests):
    datatype = 'xarray'
    data_type = xr.Dataset
    __test__ = True

    def init_data(self):
        self.image = Image((self.xs, self.ys, self.array))
        self.image_inv = Image((self.xs[::-1], self.ys[::-1], self.array[::-1, ::-1]))

    def test_dataarray_dimension_order(self):
        x = np.linspace(-3, 7, 53)
        y = np.linspace(-5, 8, 89)
        z = np.exp(-1 * (x ** 2 + y[:, np.newaxis] ** 2))
        array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
        img = Image(array)
        self.assertEqual(img.kdims, [Dimension('x'), Dimension('y')])

    def test_dataarray_shape(self):
        x = np.linspace(-3, 7, 53)
        y = np.linspace(-5, 8, 89)
        z = np.exp(-1 * (x ** 2 + y[:, np.newaxis] ** 2))
        array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
        img = Image(array, ['x', 'y'])
        self.assertEqual(img.interface.shape(img, gridded=True), (53, 89))

    def test_dataarray_shape_transposed(self):
        x = np.linspace(-3, 7, 53)
        y = np.linspace(-5, 8, 89)
        z = np.exp(-1 * (x ** 2 + y[:, np.newaxis] ** 2))
        array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
        img = Image(array, ['y', 'x'])
        self.assertEqual(img.interface.shape(img, gridded=True), (89, 53))

    def test_select_on_transposed_dataarray(self):
        x = np.linspace(-3, 7, 53)
        y = np.linspace(-5, 8, 89)
        z = np.exp(-1 * (x ** 2 + y[:, np.newaxis] ** 2))
        array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
        img = Image(array)[1:3]
        self.assertEqual(img['z'], Image(array.sel(x=slice(1, 3)))['z'])

    def test_dataarray_with_no_coords(self):
        expected_xs = list(range(2))
        expected_ys = list(range(3))
        zs = np.arange(6).reshape(2, 3)
        xrarr = xr.DataArray(zs, dims=('x', 'y'))
        img = Image(xrarr)
        self.assertTrue(all(img.data.x == expected_xs))
        self.assertTrue(all(img.data.y == expected_ys))
        img = Image(xrarr, kdims=['x', 'y'])
        self.assertTrue(all(img.data.x == expected_xs))
        self.assertTrue(all(img.data.y == expected_ys))

    def test_dataarray_with_some_coords(self):
        xs = [4.2, 1]
        zs = np.arange(6).reshape(2, 3)
        xrarr = xr.DataArray(zs, dims=('x', 'y'), coords={'x': xs})
        with self.assertRaises(ValueError):
            Image(xrarr)
        with self.assertRaises(ValueError):
            Image(xrarr, kdims=['x', 'y'])