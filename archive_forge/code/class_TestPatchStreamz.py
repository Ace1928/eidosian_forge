from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
class TestPatchStreamz(TestCase):

    def setUp(self):
        try:
            import streamz
        except:
            raise SkipTest('streamz not available')
        import hvplot.streamz

    def test_streamz_dataframe_patched(self):
        from streamz.dataframe import Random
        random_df = Random()
        self.assertIsInstance(random_df.hvplot, hvPlotTabular)

    def test_streamz_series_patched(self):
        from streamz.dataframe import Random
        random_df = Random()
        self.assertIsInstance(random_df.x.hvplot, hvPlotTabular)

    def test_streamz_dataframes_patched(self):
        from streamz.dataframe import Random
        random_df = Random()
        self.assertIsInstance(random_df.groupby('x').sum().hvplot, hvPlotTabular)

    def test_streamz_seriess_patched(self):
        from streamz.dataframe import Random
        random_df = Random()
        self.assertIsInstance(random_df.groupby('x').sum().y.hvplot, hvPlotTabular)