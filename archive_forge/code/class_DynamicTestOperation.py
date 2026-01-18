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
class DynamicTestOperation(ComparisonTestCase):

    def test_dynamic_operation(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])
        dmap_with_fn = Dynamic(dmap, operation=lambda x: x.clone(x.data * 2))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0, 5) * 2))

    def test_dynamic_operation_on_hmap(self):
        hmap = HoloMap({i: Image(sine_array(0, i)) for i in range(10)})
        dmap = Dynamic(hmap, operation=lambda x: x)
        self.assertEqual(dmap.kdims[0].name, hmap.kdims[0].name)
        self.assertEqual(dmap.kdims[0].values, hmap.keys())

    def test_dynamic_operation_link_inputs_not_transferred_on_clone(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])
        dmap_with_fn = Dynamic(dmap, link_inputs=False, operation=lambda x: x.clone(x.data * 2))
        self.assertTrue(dmap_with_fn.clone().callback.link_inputs)

    def test_dynamic_operation_on_element(self):
        img = Image(sine_array(0, 5))
        posxy = PointerXY(x=2, y=1)
        dmap_with_fn = Dynamic(img, operation=lambda obj, x, y: obj.clone(obj.data * x + y), streams=[posxy])
        element = dmap_with_fn[()]
        self.assertEqual(element, Image(sine_array(0, 5) * 2 + 1))
        self.assertEqual(dmap_with_fn.streams, [posxy])

    def test_dynamic_operation_on_element_dict(self):
        img = Image(sine_array(0, 5))
        posxy = PointerXY(x=3, y=1)
        dmap_with_fn = Dynamic(img, operation=lambda obj, x, y: obj.clone(obj.data * x + y), streams=dict(x=posxy.param.x, y=posxy.param.y))
        element = dmap_with_fn[()]
        self.assertEqual(element, Image(sine_array(0, 5) * 3 + 1))

    def test_dynamic_operation_with_kwargs(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])

        def fn(x, multiplier=2):
            return x.clone(x.data * multiplier)
        dmap_with_fn = Dynamic(dmap, operation=fn, kwargs=dict(multiplier=3))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0, 5) * 3))

    def test_dynamic_operation_init_renamed_stream_params(self):
        img = Image(sine_array(0, 5))
        stream = RangeX(rename={'x_range': 'bin_range'})
        histogram(img, bin_range=(0, 1), streams=[stream], dynamic=True)
        self.assertEqual(stream.x_range, (0, 1))

    def test_dynamic_operation_init_stream_params(self):
        img = Image(sine_array(0, 5))
        stream = Stream.define('TestStream', bin_range=None)()
        histogram(img, bin_range=(0, 1), streams=[stream], dynamic=True)
        self.assertEqual(stream.bin_range, (0, 1))