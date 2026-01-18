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
class TestStreamsDefine(ComparisonTestCase):

    def setUp(self):
        self.XY = Stream.define('XY', x=0.0, y=5.0)
        self.TypesTest = Stream.define('TypesTest', t=True, u=0, v=1.2, w=(1, 'a'), x='string', y=[], z=np.array([1, 2, 3]))
        test_param = param.Integer(default=42, doc='Test docstring')
        self.ExplicitTest = Stream.define('ExplicitTest', test=test_param)

    def test_XY_types(self):
        self.assertEqual(isinstance(self.XY.param['x'], param.Number), True)
        self.assertEqual(isinstance(self.XY.param['y'], param.Number), True)

    def test_XY_defaults(self):
        self.assertEqual(self.XY.param['x'].default, 0.0)
        self.assertEqual(self.XY.param['y'].default, 5.0)

    def test_XY_instance(self):
        xy = self.XY(x=1, y=2)
        self.assertEqual(xy.x, 1)
        self.assertEqual(xy.y, 2)

    def test_XY_set_invalid_class_x(self):
        if Version(param.__version__) > Version('2.0.0a2'):
            regexp = "Number parameter 'XY.x' only takes numeric values"
        else:
            regexp = "Parameter 'x' only takes numeric values"
        with self.assertRaisesRegex(ValueError, regexp):
            self.XY.x = 'string'

    def test_XY_set_invalid_class_y(self):
        if Version(param.__version__) > Version('2.0.0a2'):
            regexp = "Number parameter 'XY.y' only takes numeric values"
        else:
            regexp = "Parameter 'y' only takes numeric values"
        with self.assertRaisesRegex(ValueError, regexp):
            self.XY.y = 'string'

    def test_XY_set_invalid_instance_x(self):
        xy = self.XY(x=1, y=2)
        if Version(param.__version__) > Version('2.0.0a2'):
            regexp = "Number parameter 'XY.x' only takes numeric values"
        else:
            regexp = "Parameter 'x' only takes numeric values"
        with self.assertRaisesRegex(ValueError, regexp):
            xy.x = 'string'

    def test_XY_set_invalid_instance_y(self):
        xy = self.XY(x=1, y=2)
        if Version(param.__version__) > Version('2.0.0a2'):
            regexp = "Number parameter 'XY.y' only takes numeric values"
        else:
            regexp = "Parameter 'y' only takes numeric values"
        with self.assertRaisesRegex(ValueError, regexp):
            xy.y = 'string'

    def test_XY_subscriber_triggered(self):

        class Inner:

            def __init__(self):
                self.state = None

            def __call__(self, x, y):
                self.state = (x, y)
        inner = Inner()
        xy = self.XY(x=1, y=2)
        xy.add_subscriber(inner)
        xy.event(x=42, y=420)
        self.assertEqual(inner.state, (42, 420))

    def test_custom_types(self):
        self.assertEqual(isinstance(self.TypesTest.param['t'], param.Boolean), True)
        self.assertEqual(isinstance(self.TypesTest.param['u'], param.Integer), True)
        self.assertEqual(isinstance(self.TypesTest.param['v'], param.Number), True)
        self.assertEqual(isinstance(self.TypesTest.param['w'], param.Tuple), True)
        self.assertEqual(isinstance(self.TypesTest.param['x'], param.String), True)
        self.assertEqual(isinstance(self.TypesTest.param['y'], param.List), True)
        self.assertEqual(isinstance(self.TypesTest.param['z'], param.Array), True)

    def test_explicit_parameter(self):
        self.assertEqual(isinstance(self.ExplicitTest.param['test'], param.Integer), True)
        self.assertEqual(self.ExplicitTest.param['test'].default, 42)
        self.assertEqual(self.ExplicitTest.param['test'].doc, 'Test docstring')