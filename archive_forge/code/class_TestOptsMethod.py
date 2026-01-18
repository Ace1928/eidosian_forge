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
class TestOptsMethod(ComparisonTestCase):

    def setUp(self):
        self.backend = 'matplotlib'
        Store.set_current_backend(self.backend)
        self.store_copy = OptionTree(sorted(Store.options().items()), groups=Options._option_groups)
        super().setUp()

    def tearDown(self):
        Store.options(val=self.store_copy)
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}
        super().tearDown()

    def lookup_options(self, obj, group):
        return Store.lookup_options(self.backend, obj, group)

    def test_simple_clone_disabled(self):
        im = Image(np.random.rand(10, 10))
        styled_im = im.opts(interpolation='nearest', cmap='jet', clone=False)
        self.assertEqual(self.lookup_options(im, 'plot').options, {})
        self.assertEqual(self.lookup_options(styled_im, 'plot').options, {})
        assert styled_im is im
        self.assertEqual(self.lookup_options(im, 'style').options, {'cmap': 'jet', 'interpolation': 'nearest'})

    def test_simple_opts_clone_enabled(self):
        im = Image(np.random.rand(10, 10))
        styled_im = im.opts(interpolation='nearest', cmap='jet', clone=True)
        self.assertEqual(self.lookup_options(im, 'plot').options, {})
        self.assertEqual(self.lookup_options(styled_im, 'plot').options, {})
        assert styled_im is not im
        im_lookup = self.lookup_options(im, 'style').options
        self.assertEqual(im_lookup['cmap'] == 'jet', False)
        styled_im_lookup = self.lookup_options(styled_im, 'style').options
        self.assertEqual(styled_im_lookup['cmap'] == 'jet', True)

    def test_opts_method_with_utility(self):
        im = Image(np.random.rand(10, 10))
        imopts = opts.Image(cmap='Blues')
        styled_im = im.opts(imopts)
        assert styled_im is im
        self.assertEqual(self.lookup_options(im, 'style').options, {'cmap': 'Blues', 'interpolation': 'nearest'})

    def test_opts_method_dynamicmap_grouped(self):
        dmap = DynamicMap(lambda X: Curve([1, 2, X]), kdims=['X']).redim.range(X=(0, 3))
        retval = dmap.opts(padding=1, clone=True)
        assert retval is not dmap
        self.assertEqual(self.lookup_options(retval[0], 'plot').options, {'padding': 1})

    def test_opts_clear(self):
        im = Image(np.random.rand(10, 10))
        styled_im = opts.apply_groups(im, style=dict(cmap='jet', interpolation='nearest', option1='A', option2='B'), clone=False)
        self.assertEqual(self.lookup_options(im, 'style').options, {'cmap': 'jet', 'interpolation': 'nearest', 'option1': 'A', 'option2': 'B'})
        assert styled_im is im
        cleared = im.opts.clear()
        assert cleared is im
        cleared_options = self.lookup_options(cleared, 'style').options
        self.assertEqual(not any((k in ['option1', 'option2'] for k in cleared_options.keys())), True)

    def test_opts_clear_clone(self):
        im = Image(np.random.rand(10, 10))
        styled_im = opts.apply_groups(im, style=dict(cmap='jet', interpolation='nearest', option1='A', option2='B'), clone=False)
        self.assertEqual(self.lookup_options(im, 'style').options, {'cmap': 'jet', 'interpolation': 'nearest', 'option1': 'A', 'option2': 'B'})
        assert styled_im is im
        cleared = im.opts.clear(clone=True)
        assert cleared is not im
        self.assertEqual(self.lookup_options(im, 'style').options, {'cmap': 'jet', 'interpolation': 'nearest', 'option1': 'A', 'option2': 'B'})
        cleared_options = self.lookup_options(cleared, 'style').options
        self.assertEqual(not any((k in ['option1', 'option2'] for k in cleared_options.keys())), True)