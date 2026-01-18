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
class TestCrossBackendOptions(ComparisonTestCase):
    """
    Test the style system can style a single object across backends.
    """

    def setUp(self):
        self.plotly_options = Store._options.pop('plotly', None)
        self.store_mpl = OptionTree(sorted(Store.options(backend='matplotlib').items()), groups=Options._option_groups, backend='matplotlib')
        self.store_bokeh = OptionTree(sorted(Store.options(backend='bokeh').items()), groups=Options._option_groups, backend='bokeh')
        self.clear_options()
        super().setUp()

    def clear_options(self):
        Store.options(val=OptionTree(groups=['plot', 'style'], backend='matplotlib'), backend='matplotlib')
        Store.options(val=OptionTree(groups=['plot', 'style'], backend='bokeh'), backend='bokeh')
        Store.custom_options({}, backend='matplotlib')
        Store.custom_options({}, backend='bokeh')

    def tearDown(self):
        Store.options(val=self.store_mpl, backend='matplotlib')
        Store.options(val=self.store_bokeh, backend='bokeh')
        Store.current_backend = 'matplotlib'
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}
        if self.plotly_options is not None:
            Store._options['plotly'] = self.plotly_options
        super().tearDown()

    def test_mpl_bokeh_mpl(self):
        img = Image(np.random.rand(10, 10))
        Store.current_backend = 'matplotlib'
        StoreOptions.set_options(img, style={'Image': {'cmap': 'Blues'}})
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap': 'Blues'})
        Store.current_backend = 'bokeh'
        StoreOptions.set_options(img, style={'Image': {'cmap': 'Purple'}})
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap': 'Purple'})
        Store.current_backend = 'matplotlib'
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap': 'Blues'})
        Store.current_backend = 'bokeh'
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap': 'Purple'})

    def test_mpl_bokeh_offset_mpl(self):
        img = Image(np.random.rand(10, 10))
        Store.current_backend = 'matplotlib'
        StoreOptions.set_options(img, style={'Image': {'cmap': 'Blues'}})
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap': 'Blues'})
        Store.current_backend = 'bokeh'
        img2 = Image(np.random.rand(10, 10))
        StoreOptions.set_options(img2, style={'Image': {'cmap': 'Reds'}})
        img2_opts = Store.lookup_options('bokeh', img2, 'style').options
        self.assertEqual(img2_opts, {'cmap': 'Reds'})
        StoreOptions.set_options(img, style={'Image': {'cmap': 'Purple'}})
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap': 'Purple'})
        Store.current_backend = 'matplotlib'
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap': 'Blues'})
        Store.current_backend = 'bokeh'
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap': 'Purple'})

    def test_builder_backend_switch_signature(self):
        Store.options(val=self.store_mpl, backend='matplotlib')
        Store.options(val=self.store_bokeh, backend='bokeh')
        Store.set_current_backend('bokeh')
        self.assertEqual(opts.Curve.__signature__ is not None, True)
        sigkeys = opts.Curve.__signature__.parameters
        self.assertEqual('color' in sigkeys, True)
        self.assertEqual('line_width' in sigkeys, True)
        Store.set_current_backend('matplotlib')
        self.assertEqual(opts.Curve.__signature__ is not None, True)
        sigkeys = opts.Curve.__signature__.parameters
        self.assertEqual('color' in sigkeys, True)
        self.assertEqual('linewidth' in sigkeys, True)

    def test_builder_cross_backend_validation(self):
        Store.options(val=self.store_mpl, backend='matplotlib')
        Store.options(val=self.store_bokeh, backend='bokeh')
        Store.set_current_backend('bokeh')
        opts.Curve(line_dash='dotted')
        opts.Curve(linewidth=10)
        err = "In opts.Curve(...), keywords supplied are mixed across backends. Keyword(s) 'linewidth' are invalid for bokeh, 'line_dash' are invalid for matplotlib"
        with pytest.raises(ValueError) as excinfo:
            opts.Curve(linewidth=10, line_dash='dotted')
        assert err in str(excinfo.value)
        err = "In opts.Curve(...), unexpected option 'foobar' for Curve type across all extensions. Similar options for current extension ('bokeh') are: ['toolbar']."
        with pytest.raises(ValueError) as excinfo:
            opts.Curve(foobar=3)
        assert err in str(excinfo.value)
        Store.set_current_backend('matplotlib')
        err = "In opts.Curve(...), unexpected option 'foobar' for Curve type across all extensions. No similar options found."
        with pytest.raises(ValueError) as excinfo:
            opts.Curve(foobar=3)
        assert err in str(excinfo.value)