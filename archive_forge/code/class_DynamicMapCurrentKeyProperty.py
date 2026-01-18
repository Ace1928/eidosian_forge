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
class DynamicMapCurrentKeyProperty(ComparisonTestCase):

    def test_current_key_None_on_init(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=[Dimension('dim', range=(0, 10))])
        self.assertIsNone(dmap.current_key)

    def test_current_key_one_dimension(self):
        fn = lambda i: Image(sine_array(0, i))
        dmap = DynamicMap(fn, kdims=[Dimension('dim', range=(0, 10))])
        dmap[0]
        self.assertEqual(dmap.current_key, 0)
        dmap[1]
        self.assertEqual(dmap.current_key, 1)
        dmap[0]
        self.assertEqual(dmap.current_key, 0)
        self.assertNotEqual(dmap.current_key, dmap.last_key)

    def test_current_key_multiple_dimensions(self):
        fn = lambda i, j: Curve([i, j])
        dmap = DynamicMap(fn, kdims=[Dimension('i', range=(0, 5)), Dimension('j', range=(0, 5))])
        dmap[0, 2]
        self.assertEqual(dmap.current_key, (0, 2))
        dmap[5, 5]
        self.assertEqual(dmap.current_key, (5, 5))
        dmap[0, 2]
        self.assertEqual(dmap.current_key, (0, 2))
        self.assertNotEqual(dmap.current_key, dmap.last_key)