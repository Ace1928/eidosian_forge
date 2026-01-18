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
class ImageElement_GridInterfaceTests(BaseImageElementInterfaceTests):
    datatype = 'grid'
    data_type = dict
    __test__ = True

    def init_data(self):
        self.image = Image((self.xs, self.ys, self.array))
        self.image_inv = Image((self.xs[::-1], self.ys[::-1], self.array[::-1, ::-1]))

    def test_init_data_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10)
        Image((xs, self.ys, self.array))

    def test_init_data_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10)
        Image((self.xs, ys, self.array))

    def test_init_bounds_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10)
        image = Image((xs, self.ys, self.array))
        self.assertEqual(image.bounds.lbrt(), (start, 0, end, 10))

    def test_init_bounds_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10)
        image = Image((self.xs, ys, self.array))
        self.assertEqual(image.bounds.lbrt(), (-10, start, 10, end))

    def test_init_densities_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10)
        image = Image((xs, self.ys, self.array))
        self.assertEqual(image.xdensity, 1e-05)
        self.assertEqual(image.ydensity, 1)

    def test_init_densities_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10)
        image = Image((self.xs, ys, self.array))
        self.assertEqual(image.xdensity, 0.5)
        self.assertEqual(image.ydensity, 1e-05)

    def test_sample_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10)
        image = Image((xs, self.ys, self.array))
        curve = image.sample(x=xs[3])
        self.assertEqual(curve, Curve((self.ys, self.array[:, 3]), 'y', 'z'))

    def test_sample_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10)
        image = Image((self.xs, ys, self.array))
        curve = image.sample(y=ys[3])
        self.assertEqual(curve, Curve((self.xs, self.array[3]), 'x', 'z'))

    def test_range_datetime_xdim(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10)
        image = Image((xs, self.ys, self.array))
        self.assertEqual(image.range(0), (start, end))

    def test_range_datetime_ydim(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10)
        image = Image((self.xs, ys, self.array))
        self.assertEqual(image.range(1), (start, end))

    def test_dimension_values_datetime_xcoords(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10)
        image = Image((xs, self.ys, self.array))
        self.assertEqual(image.dimension_values(0, expanded=False), xs)

    def test_dimension_values_datetime_ycoords(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10)
        image = Image((self.xs, ys, self.array))
        self.assertEqual(image.dimension_values(1, expanded=False), ys)

    def test_slice_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10)
        image = Image((xs, self.ys, self.array))
        sliced = image[start + np.timedelta64(530, 'ms'):start + np.timedelta64(770, 'ms')]
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[:, 5:8])

    def test_slice_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10)
        image = Image((self.xs, ys, self.array))
        sliced = image[:, start + np.timedelta64(120, 'ms'):start + np.timedelta64(520, 'ms')]
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[1:5, :])

    def test_slice_xaxis_inv(self):
        sliced = self.image_inv[0.3:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 0, 6, 10))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[:, 5:8])

    def test_slice_yaxis_inv(self):
        sliced = self.image_inv[:, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (-10, 1.0, 10, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[1:5, :])

    def test_slice_both_axes_inv(self):
        sliced = self.image_inv[0.3:5.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 1.0, 6, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[1:5, 5:8])

    def test_slice_x_index_y_inv(self):
        sliced = self.image_inv[0.3:5.2, 5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 5.0, 6.0, 6.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[5:6, 5:8])

    def test_index_x_slice_y_inv(self):
        sliced = self.image_inv[3.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (2.0, 1.0, 4.0, 5.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[1:5, 6:7])