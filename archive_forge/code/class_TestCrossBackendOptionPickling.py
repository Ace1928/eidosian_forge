import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
class TestCrossBackendOptionPickling(TestCrossBackendOptions):
    cleanup = ['test_raw_pickle.pkl', 'test_pickle_mpl_bokeh.pkl']

    def setUp(self):
        super().setUp()
        self.raw = Image(np.random.rand(10, 10))
        Store.current_backend = 'matplotlib'
        StoreOptions.set_options(self.raw, style={'Image': {'cmap': 'Blues'}})
        Store.current_backend = 'bokeh'
        StoreOptions.set_options(self.raw, style={'Image': {'cmap': 'Purple'}})

    def tearDown(self):
        super().tearDown()
        for f in self.cleanup:
            try:
                os.remove(f)
            except Exception:
                pass

    def test_raw_pickle(self):
        """
        Test usual pickle saving and loading (no style information preserved)
        """
        fname = 'test_raw_pickle.pkl'
        with open(fname, 'wb') as handle:
            pickle.dump(self.raw, handle)
        self.clear_options()
        with open(fname, 'rb') as handle:
            img = pickle.load(handle)
        self.assertEqual(self.raw, img)
        pickle.current_backend = 'matplotlib'
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {})
        Store.current_backend = 'bokeh'
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {})

    def test_pickle_mpl_bokeh(self):
        """
        Test pickle saving and loading with Store (style information preserved)
        """
        fname = 'test_pickle_mpl_bokeh.pkl'
        with open(fname, 'wb') as handle:
            Store.dump(self.raw, handle)
        self.clear_options()
        with open(fname, 'rb') as handle:
            img = Store.load(handle)
        self.assertEqual(self.raw, img)
        Store.current_backend = 'matplotlib'
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap': 'Blues'})
        Store.current_backend = 'bokeh'
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap': 'Purple'})