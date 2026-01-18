import datetime as dt
from itertools import product
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from holoviews.element import HSV, RGB, Curve, Image
from holoviews.util.transform import dim
from .base import (
from .test_imageinterface import (
class DaskGridInterfaceTests(GridInterfaceTests):

    def setUp(self):
        if da is None:
            raise SkipTest('DaskGridInterfaceTests requires dask.')
        super().setUp()

    def init_column_data(self):
        self.xs = np.arange(11)
        self.xs_2 = self.xs ** 2
        self.y_ints = da.from_array(self.xs * 2, 3)
        self.dataset_hm = self.element((self.xs, self.y_ints), ['x'], ['y'])
        self.dataset_hm_alias = self.element((self.xs, self.y_ints), [('x', 'X')], [('y', 'Y')])

    def init_grid_data(self):
        import dask.array as da
        self.grid_xs = np.array([0, 1])
        self.grid_ys = np.array([0.1, 0.2, 0.3])
        self.grid_zs = da.from_array(np.array([[0, 1], [2, 3], [4, 5]]), 3)
        self.dataset_grid = self.element((self.grid_xs, self.grid_ys, self.grid_zs), ['x', 'y'], ['z'])
        self.dataset_grid_alias = self.element((self.grid_xs, self.grid_ys, self.grid_zs), [('x', 'X'), ('y', 'Y')], [('z', 'Z')])
        self.dataset_grid_inv = self.element((self.grid_xs[::-1], self.grid_ys[::-1], self.grid_zs), ['x', 'y'], ['z'])

    def test_dataset_array_hm(self):
        self.assertEqual(self.dataset_hm.array(), np.column_stack([self.xs, self.y_ints.compute()]))

    def test_dataset_array_hm_alias(self):
        self.assertEqual(self.dataset_hm_alias.array(), np.column_stack([self.xs, self.y_ints.compute()]))

    def test_select_lazy(self):
        import dask.array as da
        arr = da.from_array(np.arange(1, 12), 3)
        ds = Dataset({'x': range(11), 'y': arr}, 'x', 'y')
        self.assertIsInstance(ds.select(x=(0, 5)).data['y'], da.Array)

    def test_dataset_add_dimensions_values_hm(self):
        arr = da.from_array(np.arange(1, 12), 3)
        table = self.dataset_hm.add_dimension('z', 1, arr, vdim=True)
        self.assertEqual(table.vdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.arange(1, 12))

    def test_dataset_add_dimensions_values_hm_alias(self):
        arr = da.from_array(np.arange(1, 12), 3)
        table = self.dataset_hm.add_dimension(('z', 'Z'), 1, arr, vdim=True)
        self.assertEqual(table.vdims[1], 'Z')
        self.compare_arrays(table.dimension_values('Z'), np.arange(1, 12))

    def test_dataset_2D_columnar_shape(self):
        array = da.from_array(np.random.rand(11, 11), 3)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.shape, (11 * 11, 3))

    def test_dataset_2D_gridded_shape(self):
        array = da.from_array(np.random.rand(12, 11), 3)
        dataset = Dataset({'x': self.xs, 'y': range(12), 'z': array}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.interface.shape(dataset, gridded=True), (12, 11))

    def test_dataset_2D_aggregate_partial_hm(self):
        array = da.from_array(np.random.rand(11, 11), 3)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean), Dataset({'x': self.xs, 'z': np.mean(array, axis=0).compute()}, kdims=['x'], vdims=['z']))

    def test_dataset_2D_aggregate_partial_hm_alias(self):
        array = da.from_array(np.random.rand(11, 11), 3)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(dataset.aggregate(['X'], np.mean), Dataset({'x': self.xs, 'z': np.mean(array, axis=0).compute()}, kdims=[('x', 'X')], vdims=[('z', 'Z')]))

    def test_dataset_2D_reduce_hm(self):
        array = da.from_array(np.random.rand(11, 11), 3)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.reduce(['x', 'y'], np.mean), np.mean(array).compute())

    def test_dataset_2D_reduce_hm_alias(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(np.array(dataset.reduce(['x', 'y'], np.mean)), np.mean(array))
        self.assertEqual(np.array(dataset.reduce(['X', 'Y'], np.mean)), np.mean(array))

    def test_dataset_groupby_dynamic(self):
        array = da.from_array(np.random.rand(11, 11), 3)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=['x', 'y'], vdims=['z'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], dataset):
            grouped = dataset.groupby('x', dynamic=True)
        first = Dataset({'y': self.y_ints, 'z': array[:, 0]}, kdims=['y'], vdims=['z'])
        self.assertEqual(grouped[0], first)

    def test_dataset_groupby_dynamic_alias(self):
        array = da.from_array(np.random.rand(11, 11), 3)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], dataset):
            grouped = dataset.groupby('X', dynamic=True)
        first = Dataset({'y': self.y_ints, 'z': array[:, 0].compute()}, kdims=[('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(grouped[0], first)

    def test_dataset_groupby_multiple_dims(self):
        dataset = Dataset((range(8), range(8), range(8), range(8), da.from_array(np.random.rand(8, 8, 8, 8), 4)), kdims=['a', 'b', 'c', 'd'], vdims=['Value'])
        grouped = dataset.groupby(['c', 'd'])
        keys = list(product(range(8), range(8)))
        self.assertEqual(list(grouped.keys()), keys)
        for c, d in keys:
            self.assertEqual(grouped[c, d], dataset.select(c=c, d=d).reindex(['a', 'b']))

    def test_dataset_groupby_drop_dims(self):
        array = da.from_array(np.random.rand(3, 20, 10), 3)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array}, kdims=['x', 'y', 'z'], vdims=['Val'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
            partial = ds.to(Dataset, kdims=['x'], vdims=['Val'], groupby='y')
        self.assertEqual(partial.last['Val'], array[:, -1, :].T.flatten().compute())

    def test_dataset_groupby_drop_dims_dynamic(self):
        array = da.from_array(np.random.rand(3, 20, 10), 3)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array}, kdims=['x', 'y', 'z'], vdims=['Val'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
            partial = ds.to(Dataset, kdims=['x'], vdims=['Val'], groupby='y', dynamic=True)
            self.assertEqual(partial[19]['Val'], array[:, -1, :].T.flatten().compute())

    def test_dataset_groupby_drop_dims_with_vdim(self):
        array = da.from_array(np.random.rand(3, 20, 10), 3)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array, 'Val2': array * 2}, kdims=['x', 'y', 'z'], vdims=['Val', 'Val2'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
            partial = ds.to(Dataset, kdims=['Val'], vdims=['Val2'], groupby='y')
        self.assertEqual(partial.last['Val'], array[:, -1, :].T.flatten().compute())

    def test_dataset_groupby_drop_dims_dynamic_with_vdim(self):
        array = da.from_array(np.random.rand(3, 20, 10), 3)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array, 'Val2': array * 2}, kdims=['x', 'y', 'z'], vdims=['Val', 'Val2'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
            partial = ds.to(Dataset, kdims=['Val'], vdims=['Val2'], groupby='y', dynamic=True)
            self.assertEqual(partial[19]['Val'], array[:, -1, :].T.flatten().compute())

    def test_dataset_get_dframe(self):
        df = self.dataset_hm.dframe()
        self.assertEqual(df.x.values, self.xs)
        self.assertEqual(df.y.values, self.y_ints.compute())