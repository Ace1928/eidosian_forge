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
class DynamicMapUnboundedProperty(ComparisonTestCase):

    def test_callable_bounded_init(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=[Dimension('dim', range=(0, 10))])
        self.assertEqual(dmap.unbounded, [])

    def test_callable_bounded_clone(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=[Dimension('dim', range=(0, 10))])
        self.assertEqual(dmap, dmap.clone())
        self.assertEqual(dmap.unbounded, [])

    def test_sampled_unbounded_init(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])
        self.assertEqual(dmap.unbounded, ['i'])

    def test_sampled_unbounded_resample(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=['i'])
        self.assertEqual(dmap[{0, 1, 2}].keys(), [0, 1, 2])
        self.assertEqual(dmap.unbounded, ['i'])

    def test_mixed_kdim_streams_unbounded(self):
        dmap = DynamicMap(lambda x, y, z: x + y, kdims=['z'], streams=[XY()])
        self.assertEqual(dmap.unbounded, ['z'])

    def test_mixed_kdim_streams_bounded_redim(self):
        dmap = DynamicMap(lambda x, y, z: x + y, kdims=['z'], streams=[XY()])
        self.assertEqual(dmap.redim.range(z=(-0.5, 0.5)).unbounded, [])