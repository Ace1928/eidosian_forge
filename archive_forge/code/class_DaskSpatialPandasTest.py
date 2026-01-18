from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests
class DaskSpatialPandasTest(GeomTests, RoundTripTests):
    """
    Test of the DaskSpatialPandasInterface.
    """
    datatype = 'dask_spatialpandas'
    interface = DaskSpatialPandasInterface
    __test__ = True

    def setUp(self):
        if spatialpandas is None:
            raise SkipTest('DaskSpatialPandasInterface requires spatialpandas, skipping tests')
        elif dd is None:
            raise SkipTest('DaskSpatialPandasInterface requires dask, skipping tests')
        super(GeomTests, self).setUp()

    def test_array_points_iloc_index_row(self):
        raise SkipTest('Not supported')

    def test_array_points_iloc_index_rows(self):
        raise SkipTest('Not supported')

    def test_array_points_iloc_index_rows_index_cols(self):
        raise SkipTest('Not supported')

    def test_array_points_iloc_slice_rows(self):
        raise SkipTest('Not supported')

    def test_array_points_iloc_slice_rows_no_start(self):
        raise SkipTest('Not supported')

    def test_array_points_iloc_slice_rows_no_end(self):
        raise SkipTest('Not supported')

    def test_array_points_iloc_slice_rows_no_stop(self):
        raise SkipTest('Not supported')

    def test_multi_polygon_iloc_index_row(self):
        raise SkipTest('Not supported')

    def test_multi_polygon_iloc_index_rows(self):
        raise SkipTest('Not supported')

    def test_multi_polygon_iloc_slice_rows(self):
        raise SkipTest('Not supported')

    def test_dict_dataset_add_dimension_values(self):
        raise SkipTest('Not supported')

    def test_sort_by_value(self):
        raise SkipTest('Not supported')