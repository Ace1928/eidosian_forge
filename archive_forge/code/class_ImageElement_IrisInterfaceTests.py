from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset, concat
from geoviews.data.iris import coord_to_dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Image
from holoviews.tests.core.data.test_imageinterface import BaseImageElementInterfaceTests
from holoviews.tests.core.data.test_gridinterface import BaseGridInterfaceTests
class ImageElement_IrisInterfaceTests(BaseImageElementInterfaceTests):
    datatype = 'cube'
    __test__ = True

    def init_data(self):
        xs = np.linspace(-9, 9, 10)
        ys = np.linspace(0.5, 9.5, 10)
        self.xs = xs
        self.ys = ys
        self.array = np.arange(10) * np.arange(10)[:, np.newaxis]
        self.image = Image((xs, ys, self.array))
        self.image_inv = Image((xs[::-1], ys[::-1], self.array[::-1, ::-1]))

    def test_init_data_datetime_xaxis(self):
        raise SkipTest('Not supported')

    def test_init_data_datetime_yaxis(self):
        raise SkipTest('Not supported')

    def test_init_bounds_datetime_xaxis(self):
        raise SkipTest('Not supported')

    def test_init_bounds_datetime_yaxis(self):
        raise SkipTest('Not supported')

    def test_init_densities_datetime_xaxis(self):
        raise SkipTest('Not supported')

    def test_init_densities_datetime_yaxis(self):
        raise SkipTest('Not supported')

    def test_range_datetime_xdim(self):
        raise SkipTest('Not supported')

    def test_range_datetime_ydim(self):
        raise SkipTest('Not supported')

    def test_dimension_values_datetime_xcoords(self):
        raise SkipTest('Not supported')

    def test_dimension_values_datetime_ycoords(self):
        raise SkipTest('Not supported')

    def test_slice_datetime_xaxis(self):
        raise SkipTest('Not supported')

    def test_slice_datetime_yaxis(self):
        raise SkipTest('Not supported')

    def test_reduce_to_scalar(self):
        raise SkipTest('Not supported')

    def test_reduce_x_dimension(self):
        raise SkipTest('Not supported')

    def test_reduce_y_dimension(self):
        raise SkipTest('Not supported')

    def test_aggregate_with_spreadfn(self):
        raise SkipTest('Not supported')

    def test_sample_datetime_xaxis(self):
        raise SkipTest('Not supported')

    def test_sample_datetime_yaxis(self):
        raise SkipTest('Not supported')

    def test_sample_coords(self):
        raise SkipTest('Not supported')