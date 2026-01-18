from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
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