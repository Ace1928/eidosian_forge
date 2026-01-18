from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
@pytest.mark.parametrize('geometry', [geometry_encl, geometry_noencl])
class TestSpatialSelectColumnarDaskMeta:

    @dd_available
    def test_meta_dtype(self, geometry, dask_df, _method):
        mask = spatial_select_columnar(dask_df.x, dask_df.y, geometry, _method)
        assert mask._meta.dtype == np.bool_