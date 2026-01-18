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
class TestStoreInheritance(ComparisonTestCase):
    """
    Tests to prevent regression after fix in 71c1f3a that resolves
    issue #43
    """

    def setUp(self):
        self.backend = 'matplotlib'
        Store.set_current_backend(self.backend)
        self.store_copy = OptionTree(sorted(Store.options().items()), groups=Options._option_groups)
        Store.options(val=OptionTree(groups=['plot', 'style'], backend=self.backend))
        options = Store.options()
        self.default_plot = dict(plot1='plot1', plot2='plot2')
        options.Histogram = Options('plot', **self.default_plot)
        self.default_style = dict(style1='style1', style2='style2')
        options.Histogram = Options('style', **self.default_style)
        data = [np.random.normal() for i in range(10000)]
        frequencies, edges = np.histogram(data, 20)
        self.hist = Histogram((edges, frequencies))
        super().setUp()

    def tearDown(self):
        Store.options(val=self.store_copy)
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}
        super().tearDown()

    def lookup_options(self, obj, group):
        return Store.lookup_options(self.backend, obj, group)

    def test_original_style_options(self):
        self.assertEqual(self.lookup_options(self.hist, 'style').options, self.default_style)

    def test_original_plot_options(self):
        self.assertEqual(self.lookup_options(self.hist, 'plot').options, self.default_plot)

    def test_plot_inheritance_addition(self):
        """Adding an element"""
        hist2 = opts.apply_groups(self.hist, plot={'plot3': 'plot3'})
        self.assertEqual(self.lookup_options(hist2, 'plot').options, dict(plot1='plot1', plot2='plot2', plot3='plot3'))
        self.assertEqual(self.lookup_options(hist2, 'style').options, self.default_style)

    def test_plot_inheritance_override(self):
        """Overriding an element"""
        hist2 = opts.apply_groups(self.hist, plot={'plot1': 'plot_child'})
        self.assertEqual(self.lookup_options(hist2, 'plot').options, dict(plot1='plot_child', plot2='plot2'))
        self.assertEqual(self.lookup_options(hist2, 'style').options, self.default_style)

    def test_style_inheritance_addition(self):
        """Adding an element"""
        hist2 = opts.apply_groups(self.hist, style={'style3': 'style3'})
        self.assertEqual(self.lookup_options(hist2, 'style').options, dict(style1='style1', style2='style2', style3='style3'))
        self.assertEqual(self.lookup_options(hist2, 'plot').options, self.default_plot)

    def test_style_inheritance_override(self):
        """Overriding an element"""
        hist2 = opts.apply_groups(self.hist, style={'style1': 'style_child'})
        self.assertEqual(self.lookup_options(hist2, 'style').options, dict(style1='style_child', style2='style2'))
        self.assertEqual(self.lookup_options(hist2, 'plot').options, self.default_plot)

    def test_style_transfer(self):
        hist = opts.apply_groups(self.hist, style={'style1': 'style_child'})
        hist2 = self.hist.opts()
        opts_kwargs = Store.lookup_options('matplotlib', hist2, 'style').kwargs
        self.assertEqual(opts_kwargs, {'style1': 'style1', 'style2': 'style2'})
        Store.transfer_options(hist, hist2, 'matplotlib')
        opts_kwargs = Store.lookup_options('matplotlib', hist2, 'style').kwargs
        self.assertEqual(opts_kwargs, {'style1': 'style_child', 'style2': 'style2'})