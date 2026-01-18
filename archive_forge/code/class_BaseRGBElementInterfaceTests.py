import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
class BaseRGBElementInterfaceTests(InterfaceTests):
    element = RGB
    __test__ = False

    def init_grid_data(self):
        self.xs = np.linspace(-9, 9, 10)
        self.ys = np.linspace(0.5, 9.5, 10)
        self.rgb_array = np.random.rand(10, 10, 3)

    def init_data(self):
        self.rgb = RGB(self.rgb_array[::-1], bounds=(-10, 0, 10, 10))

    def test_init_bounds(self):
        self.assertEqual(self.rgb.bounds.lbrt(), (-10, 0, 10, 10))

    def test_init_densities(self):
        self.assertEqual(self.rgb.xdensity, 0.5)
        self.assertEqual(self.rgb.ydensity, 1)

    def test_dimension_values_xs(self):
        self.assertEqual(self.rgb.dimension_values(0, expanded=False), np.linspace(-9, 9, 10))

    def test_dimension_values_ys(self):
        self.assertEqual(self.rgb.dimension_values(1, expanded=False), np.linspace(0.5, 9.5, 10))

    def test_dimension_values_vdims(self):
        self.assertEqual(self.rgb.dimension_values(2, flat=False), self.rgb_array[:, :, 0])
        self.assertEqual(self.rgb.dimension_values(3, flat=False), self.rgb_array[:, :, 1])
        self.assertEqual(self.rgb.dimension_values(4, flat=False), self.rgb_array[:, :, 2])

    def test_slice_xaxis(self):
        sliced = self.rgb[0.3:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 0, 6, 10))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.rgb_array[:, 5:8, 0])

    def test_slice_yaxis(self):
        sliced = self.rgb[:, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (-10, 1.0, 10, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.rgb_array[1:5, :, 0])

    def test_slice_both_axes(self):
        sliced = self.rgb[0.3:5.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 1.0, 6, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.rgb_array[1:5, 5:8, 0])

    def test_slice_x_index_y(self):
        sliced = self.rgb[0.3:5.2, 5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 5.0, 6.0, 6.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.rgb_array[5:6, 5:8, 0])

    def test_index_x_slice_y(self):
        sliced = self.rgb[3.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (2.0, 1.0, 4.0, 5.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.rgb_array[1:5, 6:7, 0])

    def test_select_value_dimension_rgb(self):
        self.assertEqual(self.rgb[..., 'R'], Image(np.flipud(self.rgb_array[:, :, 0]), bounds=self.rgb.bounds, vdims=[Dimension('R', range=(0, 1))], datatype=['image']))

    def test_select_single_coordinate(self):
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.rgb):
            self.assertEqual(self.rgb[5.2, 3.1], self.rgb.clone([tuple(self.rgb_array[3, 7])], kdims=[], new_type=Dataset))

    def test_reduce_to_single_values(self):
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.rgb):
            self.assertEqual(self.rgb.reduce(['x', 'y'], function=np.mean), self.rgb.clone([tuple(np.mean(self.rgb_array, axis=(0, 1)))], kdims=[], new_type=Dataset))

    def test_sample_xcoord(self):
        ys = np.linspace(0.5, 9.5, 10)
        data = (ys,) + tuple((self.rgb_array[:, 7, i] for i in range(3)))
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.rgb):
            self.assertEqual(self.rgb.sample(x=5), self.rgb.clone(data, kdims=['y'], new_type=Curve))

    def test_sample_ycoord(self):
        xs = np.linspace(-9, 9, 10)
        data = (xs,) + tuple((self.rgb_array[4, :, i] for i in range(3)))
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.rgb):
            self.assertEqual(self.rgb.sample(y=5), self.rgb.clone(data, kdims=['x'], new_type=Curve))

    def test_dataset_reindex_constant(self):
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe', 'grid'], self.rgb):
            ds = Dataset(self.rgb.select(x=0))
            reindexed = ds.reindex(['y'], ['R'])
        data = Dataset(ds.columns(['y', 'R']), kdims=['y'], vdims=[ds.vdims[0]])
        self.assertEqual(reindexed, data)

    def test_dataset_reindex_non_constant(self):
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe', 'grid'], self.rgb):
            ds = Dataset(self.rgb)
            reindexed = ds.reindex(['y'], ['R'])
        data = Dataset(ds.columns(['y', 'R']), kdims=['y'], vdims=[ds.vdims[0]])
        self.assertEqual(reindexed, data)