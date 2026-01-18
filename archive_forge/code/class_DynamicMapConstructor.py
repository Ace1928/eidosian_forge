import time
import uuid
from collections import deque
import numpy as np
import param
import pytest
from holoviews import Dimension, GridSpace, Layout, NdLayout, NdOverlay
from holoviews.core.options import Store
from holoviews.core.spaces import Callable, DynamicMap, HoloMap
from holoviews.element import Curve, Image, Points, Scatter, Text
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import (
from holoviews.util import Dynamic
from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
class DynamicMapConstructor(ComparisonTestCase):

    def test_simple_constructor_kdims(self):
        DynamicMap(lambda x: x, kdims=['test'])

    def test_simple_constructor_invalid_no_kdims(self):
        regexp = "Callable '<lambda>' accepts more positional arguments than there are kdims and stream parameters"
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(lambda x: x)

    def test_simple_constructor_invalid(self):
        regexp = "Callback '<lambda>' signature over \\['x'\\] does not accommodate required kdims \\['x', 'y'\\]"
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(lambda x: x, kdims=['x', 'y'])

    def test_simple_constructor_streams(self):
        DynamicMap(lambda x: x, streams=[PointerX()])

    def test_simple_constructor_streams_dict(self):
        pointerx = PointerX()
        DynamicMap(lambda x: x, streams=dict(x=pointerx.param.x))

    def test_simple_constructor_streams_dict_panel_widget(self):
        import panel as pn
        DynamicMap(lambda x: x, streams=dict(x=pn.widgets.FloatSlider()))

    def test_simple_constructor_streams_dict_parameter(self):
        test = ExampleParameterized()
        DynamicMap(lambda x: x, streams=dict(x=test.param.example))

    def test_simple_constructor_streams_dict_class_parameter(self):
        DynamicMap(lambda x: x, streams=dict(x=ExampleParameterized.param.example))

    def test_simple_constructor_streams_dict_invalid(self):
        regexp = 'Cannot handle value 3 in streams dictionary'
        with self.assertRaisesRegex(TypeError, regexp):
            DynamicMap(lambda x: x, streams=dict(x=3))

    def test_simple_constructor_streams_invalid_uninstantiated(self):
        regexp = 'The supplied streams list contains objects that are not Stream instances:(.+?)'
        with self.assertRaisesRegex(TypeError, regexp):
            DynamicMap(lambda x: x, streams=[PointerX])

    def test_simple_constructor_streams_invalid_type(self):
        regexp = 'The supplied streams list contains objects that are not Stream instances:(.+?)'
        with self.assertRaisesRegex(TypeError, regexp):
            DynamicMap(lambda x: x, streams=[3])

    def test_simple_constructor_streams_invalid_mismatch(self):
        regexp = "Callable '<lambda>' missing keywords to accept stream parameters: y"
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(lambda x: x, streams=[PointerXY()])

    def test_simple_constructor_positional_stream_args(self):
        DynamicMap(lambda v: v, streams=[PointerXY()], positional_stream_args=True)

    def test_simple_constructor_streams_invalid_mismatch_named(self):

        def foo(x):
            return x
        regexp = "Callable 'foo' missing keywords to accept stream parameters: y"
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(foo, streams=[PointerXY()])