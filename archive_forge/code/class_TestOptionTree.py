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
class TestOptionTree(ComparisonTestCase):

    def setUp(self):
        super().setUp()
        self.original_option_groups = Options._option_groups[:]
        Options._option_groups = ['group1', 'group2']

    def tearDown(self):
        Options._option_groups = self.original_option_groups
        super().tearDown()

    def test_optiontree_init_1(self):
        OptionTree(groups=['group1', 'group2'])

    def test_optiontree_init_2(self):
        OptionTree(groups=['group1', 'group2'])

    def test_optiontree_setter_getter(self):
        options = OptionTree(groups=['group1', 'group2'])
        opts = Options('group1', kw1='value')
        options.MyType = opts
        self.assertEqual(options.MyType['group1'], opts)
        self.assertEqual(options.MyType['group1'].options, {'kw1': 'value'})

    def test_optiontree_dict_setter_getter(self):
        options = OptionTree(groups=['group1', 'group2'])
        opts1 = Options(kw1='value1')
        opts2 = Options(kw2='value2')
        options.MyType = {'group1': opts1, 'group2': opts2}
        self.assertEqual(options.MyType['group1'], opts1)
        self.assertEqual(options.MyType['group1'].options, {'kw1': 'value1'})
        self.assertEqual(options.MyType['group2'], opts2)
        self.assertEqual(options.MyType['group2'].options, {'kw2': 'value2'})

    def test_optiontree_inheritance(self):
        options = OptionTree(groups=['group1', 'group2'])
        opts1 = Options(kw1='value1')
        opts2 = Options(kw2='value2')
        options.MyType = {'group1': opts1, 'group2': opts2}
        opts3 = Options(kw3='value3')
        opts4 = Options(kw4='value4')
        options.MyType.Child = {'group1': opts3, 'group2': opts4}
        self.assertEqual(options.MyType.Child.options('group1').kwargs, {'kw1': 'value1', 'kw3': 'value3'})
        self.assertEqual(options.MyType.Child.options('group2').kwargs, {'kw2': 'value2', 'kw4': 'value4'})

    def test_optiontree_inheritance_flipped(self):
        """
        Tests for ordering problems manifested in issue #93
        """
        options = OptionTree(groups=['group1', 'group2'])
        opts3 = Options(kw3='value3')
        opts4 = Options(kw4='value4')
        options.MyType.Child = {'group1': opts3, 'group2': opts4}
        opts1 = Options(kw1='value1')
        opts2 = Options(kw2='value2')
        options.MyType = {'group1': opts1, 'group2': opts2}
        self.assertEqual(options.MyType.Child.options('group1').kwargs, {'kw1': 'value1', 'kw3': 'value3'})
        self.assertEqual(options.MyType.Child.options('group2').kwargs, {'kw2': 'value2', 'kw4': 'value4'})