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
class TestStoreInheritanceDynamic(ComparisonTestCase):
    """
    Tests to prevent regression after fix in PR #646
    """

    def setUp(self):
        self.backend = 'matplotlib'
        Store.set_current_backend(self.backend)
        options = Store.options()
        self.store_copy = OptionTree(sorted(options.items()), groups=Options._option_groups, backend=options.backend)
        super().setUp()

    def tearDown(self):
        Store.options(val=self.store_copy)
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}
        super().tearDown()

    def initialize_option_tree(self):
        Store.options(val=OptionTree(groups=['plot', 'style']))
        options = Store.options()
        options.Image = Options('style', cmap='hot', interpolation='nearest')
        return options

    def test_empty_group(self):
        """Test to prevent regression of issue fixed in #5131"""
        ls = np.linspace(0, 10, 200)
        xx, yy = np.meshgrid(ls, ls)
        util.render(Image(np.sin(xx) * np.cos(yy), group='').opts(cmap='greys'))

    def test_merge_keywords(self):
        options = self.initialize_option_tree()
        options.Image = Options('style', clims=(0, 0.5))
        expected = {'clims': (0, 0.5), 'cmap': 'hot', 'interpolation': 'nearest'}
        direct_kws = options.Image.groups['style'].kwargs
        inherited_kws = options.Image.options('style').kwargs
        self.assertEqual(direct_kws, expected)
        self.assertEqual(inherited_kws, expected)

    def test_merge_keywords_disabled(self):
        options = self.initialize_option_tree()
        options.Image = Options('style', clims=(0, 0.5), merge_keywords=False)
        expected = {'clims': (0, 0.5)}
        direct_kws = options.Image.groups['style'].kwargs
        inherited_kws = options.Image.options('style').kwargs
        self.assertEqual(direct_kws, expected)
        self.assertEqual(inherited_kws, expected)

    def test_specification_general_to_specific_group(self):
        """
        Test order of specification starting with general and moving
        to specific
        """
        options = self.initialize_option_tree()
        obj = Image(np.random.rand(10, 10), group='SomeGroup')
        options.Image = Options('style', cmap='viridis')
        options.Image.SomeGroup = Options('style', alpha=0.2)
        expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
        lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(lookup.kwargs, expected)
        node1 = options.Image.groups['style']
        node2 = options.Image.SomeGroup.groups['style']
        self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
        self.assertEqual(node2.kwargs, {'alpha': 0.2})

    def test_specification_general_to_specific_group_and_label(self):
        """
        Test order of specification starting with general and moving
        to specific
        """
        options = self.initialize_option_tree()
        obj = Image(np.random.rand(10, 10), group='SomeGroup', label='SomeLabel')
        options.Image = Options('style', cmap='viridis')
        options.Image.SomeGroup.SomeLabel = Options('style', alpha=0.2)
        expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
        lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(lookup.kwargs, expected)
        node1 = options.Image.groups['style']
        node2 = options.Image.SomeGroup.SomeLabel.groups['style']
        self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
        self.assertEqual(node2.kwargs, {'alpha': 0.2})

    def test_specification_specific_to_general_group(self):
        """
        Test order of specification starting with a specific option and
        then specifying a general one
        """
        options = self.initialize_option_tree()
        options.Image.SomeGroup = Options('style', alpha=0.2)
        obj = Image(np.random.rand(10, 10), group='SomeGroup')
        options.Image = Options('style', cmap='viridis')
        expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
        lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(lookup.kwargs, expected)
        node1 = options.Image.groups['style']
        node2 = options.Image.SomeGroup.groups['style']
        self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
        self.assertEqual(node2.kwargs, {'alpha': 0.2})

    def test_specification_specific_to_general_group_and_label(self):
        """
        Test order of specification starting with general and moving
        to specific
        """
        options = self.initialize_option_tree()
        options.Image.SomeGroup.SomeLabel = Options('style', alpha=0.2)
        obj = Image(np.random.rand(10, 10), group='SomeGroup', label='SomeLabel')
        options.Image = Options('style', cmap='viridis')
        expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
        lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(lookup.kwargs, expected)
        node1 = options.Image.groups['style']
        node2 = options.Image.SomeGroup.SomeLabel.groups['style']
        self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
        self.assertEqual(node2.kwargs, {'alpha': 0.2})

    def test_backend_opts_to_default_inheritance(self):
        """
        Checks customs inheritance backs off to default tree correctly
        using .opts.
        """
        options = self.initialize_option_tree()
        options.Image.A.B = Options('style', alpha=0.2)
        obj = Image(np.random.rand(10, 10), group='A', label='B')
        expected_obj = {'alpha': 0.2, 'cmap': 'hot', 'interpolation': 'nearest'}
        obj_lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(obj_lookup.kwargs, expected_obj)
        custom_obj = opts.apply_groups(obj, style=dict(clims=(0, 0.5)))
        expected_custom_obj = dict(clims=(0, 0.5), **expected_obj)
        custom_obj_lookup = Store.lookup_options('matplotlib', custom_obj, 'style')
        self.assertEqual(custom_obj_lookup.kwargs, expected_custom_obj)

    def test_custom_magic_to_default_inheritance(self):
        """
        Checks customs inheritance backs off to default tree correctly
        simulating the %%opts cell magic.
        """
        options = self.initialize_option_tree()
        options.Image.A.B = Options('style', alpha=0.2)
        obj = Image(np.random.rand(10, 10), group='A', label='B')
        expected_obj = {'alpha': 0.2, 'cmap': 'hot', 'interpolation': 'nearest'}
        obj_lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(obj_lookup.kwargs, expected_obj)
        custom_tree = {0: OptionTree(groups=Options._option_groups, style={'Image': dict(clims=(0, 0.5))})}
        Store._custom_options['matplotlib'] = custom_tree
        obj.id = 0
        expected_custom_obj = dict(clims=(0, 0.5), **expected_obj)
        custom_obj_lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(custom_obj_lookup.kwargs, expected_custom_obj)