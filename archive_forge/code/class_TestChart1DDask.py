import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
class TestChart1DDask(TestChart1D):

    def setUp(self):
        super().setUp()
        try:
            import dask.dataframe as dd
        except:
            raise SkipTest('Dask not available')
        import hvplot.dask
        self.df = dd.from_pandas(self.df, npartitions=2)
        self.dt_df = dd.from_pandas(self.dt_df, npartitions=3)
        self.cat_df = dd.from_pandas(self.cat_df, npartitions=3)
        self.cat_only_df = dd.from_pandas(self.cat_only_df, npartitions=1)

    def test_by_datetime_accessor(self):
        raise SkipTest("Can't expand dt accessor columns when using dask")