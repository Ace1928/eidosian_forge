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
class DynamicTestOverlay(ComparisonTestCase):

    def test_dynamic_element_overlay(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])
        dynamic_overlay = dmap * Image(sine_array(0, 10))
        overlaid = Image(sine_array(0, 5)) * Image(sine_array(0, 10))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_element_underlay(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])
        dynamic_overlay = Image(sine_array(0, 10)) * dmap
        overlaid = Image(sine_array(0, 10)) * Image(sine_array(0, 5))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_dynamicmap_overlay(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])
        fn2 = lambda i: Image(sine_array(0, i * 2))
        dmap2 = DynamicMap(fn2, kdims=['i'])
        dynamic_overlay = dmap * dmap2
        overlaid = Image(sine_array(0, 5)) * Image(sine_array(0, 10))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_holomap_overlay(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])
        hmap = HoloMap({i: Image(sine_array(0, i * 2)) for i in range(10)}, kdims=['i'])
        dynamic_overlay = dmap * hmap
        overlaid = Image(sine_array(0, 5)) * Image(sine_array(0, 10))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_overlay_memoization(self):
        """Tests that Callable memoizes unchanged callbacks"""

        def fn(x, y):
            return Scatter([(x, y)])
        dmap = DynamicMap(fn, kdims=[], streams=[PointerXY()])
        counter = [0]

        def fn2(x, y):
            counter[0] += 1
            return Image(np.random.rand(10, 10))
        dmap2 = DynamicMap(fn2, kdims=[], streams=[PointerXY()])
        overlaid = dmap * dmap2
        overlay = overlaid[()]
        self.assertEqual(overlay.Scatter.I, fn(None, None))
        dmap.event(x=1, y=2)
        overlay = overlaid[()]
        self.assertEqual(overlay.Scatter.I, fn(1, 2))
        self.assertEqual(counter[0], 1)

    def test_dynamic_event_renaming_valid(self):

        def fn(x1, y1):
            return Scatter([(x1, y1)])
        xy = PointerXY(rename={'x': 'x1', 'y': 'y1'})
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        dmap.event(x1=1, y1=2)

    def test_dynamic_event_renaming_invalid(self):

        def fn(x1, y1):
            return Scatter([(x1, y1)])
        xy = PointerXY(rename={'x': 'x1', 'y': 'y1'})
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        regexp = '(.+?)do not correspond to stream parameters'
        with self.assertRaisesRegex(KeyError, regexp):
            dmap.event(x=1, y=2)