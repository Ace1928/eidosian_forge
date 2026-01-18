from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
class TestParameterRenaming(ComparisonTestCase):

    def test_simple_rename_constructor(self):
        xy = PointerXY(rename={'x': 'xtest', 'y': 'ytest'}, x=0, y=4)
        self.assertEqual(xy.contents, {'xtest': 0, 'ytest': 4})

    def test_invalid_rename_constructor(self):
        regexp = '(.+?)is not a stream parameter'
        with self.assertRaisesRegex(KeyError, regexp):
            PointerXY(rename={'x': 'xtest', 'z': 'ytest'}, x=0, y=4)
            self.assertEqual(str(cm).endswith(), True)

    def test_clashing_rename_constructor(self):
        regexp = '(.+?)parameter of the same name'
        with self.assertRaisesRegex(KeyError, regexp):
            PointerXY(rename={'x': 'xtest', 'y': 'x'}, x=0, y=4)

    def test_simple_rename_method(self):
        xy = PointerXY(x=0, y=4)
        renamed = xy.rename(x='xtest', y='ytest')
        self.assertEqual(renamed.contents, {'xtest': 0, 'ytest': 4})

    def test_invalid_rename_method(self):
        xy = PointerXY(x=0, y=4)
        regexp = '(.+?)is not a stream parameter'
        with self.assertRaisesRegex(KeyError, regexp):
            xy.rename(x='xtest', z='ytest')

    def test_clashing_rename_method(self):
        xy = PointerXY(x=0, y=4)
        regexp = '(.+?)parameter of the same name'
        with self.assertRaisesRegex(KeyError, regexp):
            xy.rename(x='xtest', y='x')

    def test_update_rename_valid(self):
        xy = PointerXY(x=0, y=4)
        renamed = xy.rename(x='xtest', y='ytest')
        renamed.event(x=4, y=8)
        self.assertEqual(renamed.contents, {'xtest': 4, 'ytest': 8})

    def test_update_rename_invalid(self):
        xy = PointerXY(x=0, y=4)
        renamed = xy.rename(y='ytest')
        regexp = "ytest' is not a parameter of(.+?)"
        with self.assertRaisesRegex(ValueError, regexp):
            renamed.event(ytest=8)

    def test_rename_suppression(self):
        renamed = PointerXY(x=0, y=0).rename(x=None)
        self.assertEqual(renamed.contents, {'y': 0})

    def test_rename_suppression_reenable(self):
        renamed = PointerXY(x=0, y=0).rename(x=None)
        self.assertEqual(renamed.contents, {'y': 0})
        reenabled = renamed.rename(x='foo')
        self.assertEqual(reenabled.contents, {'foo': 0, 'y': 0})