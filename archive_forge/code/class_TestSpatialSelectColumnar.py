from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
class TestSpatialSelectColumnar:
    __test__ = False
    method = None
    geometry_encl = np.array([[-1, 0.5], [1, 0.5], [0, -1.5], [-2, -1.5]], dtype=float)
    pt_mask_encl = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0], dtype=bool)
    geometry_noencl = np.array([[10, 10], [10, 11], [11, 11], [11, 10]], dtype=float)
    pt_mask_noencl = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

    @pytest.fixture(scope='module')
    def pandas_df(self):
        return pd.DataFrame({'x': [-1, 0, 1, -1, 0, 1, -1, 0, 1], 'y': [1, 1, 1, 0, 0, 0, -1, -1, -1]}, dtype=float)

    @pytest.fixture(scope='function')
    def dask_df(self, pandas_df):
        return dd.from_pandas(pandas_df, npartitions=2)

    @pytest.fixture(scope='function')
    def _method(self):
        return self.method

    @pytest.mark.parametrize('geometry,pt_mask', [(geometry_encl, pt_mask_encl), (geometry_noencl, pt_mask_noencl)])
    class TestSpatialSelectColumnarPtMask:

        def test_pandas(self, geometry, pt_mask, pandas_df, _method):
            mask = spatial_select_columnar(pandas_df.x, pandas_df.y, geometry, _method)
            assert np.array_equal(mask, pt_mask)

        @dd_available
        def test_dask(self, geometry, pt_mask, dask_df, _method):
            mask = spatial_select_columnar(dask_df.x, dask_df.y, geometry, _method)
            assert np.array_equal(mask.compute(), pt_mask)

        def test_numpy(self, geometry, pt_mask, pandas_df, _method):
            mask = spatial_select_columnar(pandas_df.x.to_numpy(copy=True), pandas_df.y.to_numpy(copy=True), geometry, _method)
            assert np.array_equal(mask, pt_mask)

    @pytest.mark.parametrize('geometry', [geometry_encl, geometry_noencl])
    class TestSpatialSelectColumnarDaskMeta:

        @dd_available
        def test_meta_dtype(self, geometry, dask_df, _method):
            mask = spatial_select_columnar(dask_df.x, dask_df.y, geometry, _method)
            assert mask._meta.dtype == np.bool_