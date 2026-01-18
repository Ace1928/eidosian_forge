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
class DynamicMapPositionalStreamArgs(ComparisonTestCase):

    def test_positional_stream_args_without_streams(self):
        fn = lambda i: Curve([i, i])
        dmap = DynamicMap(fn, kdims=['i'], positional_stream_args=True)
        self.assertEqual(dmap[0], Curve([0, 0]))

    def test_positional_stream_args_with_only_stream(self):
        fn = lambda s: Curve([s['x'], s['y']])
        xy_stream = XY(x=1, y=2)
        dmap = DynamicMap(fn, streams=[xy_stream], positional_stream_args=True)
        self.assertEqual(dmap[()], Curve([1, 2]))
        xy_stream.event(x=5, y=7)
        self.assertEqual(dmap[()], Curve([5, 7]))

    def test_positional_stream_args_with_single_kdim_and_stream(self):
        fn = lambda i, s: Points([i, i]) + Curve([s['x'], s['y']])
        xy_stream = XY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['i'], streams=[xy_stream], positional_stream_args=True)
        self.assertEqual(dmap[6], Points([6, 6]) + Curve([1, 2]))
        xy_stream.event(x=5, y=7)
        self.assertEqual(dmap[3], Points([3, 3]) + Curve([5, 7]))

    def test_positional_stream_args_with_multiple_kdims_and_stream(self):
        fn = lambda i, j, s1, s2: Points([i, j]) + Curve([s1['x'], s2['y']])
        x_stream = X(x=2)
        y_stream = Y(y=3)
        dmap = DynamicMap(fn, kdims=['i', 'j'], streams=[x_stream, y_stream], positional_stream_args=True)
        self.assertEqual(dmap[0, 1], Points([0, 1]) + Curve([2, 3]))
        x_stream.event(x=5)
        y_stream.event(y=6)
        self.assertEqual(dmap[3, 4], Points([3, 4]) + Curve([5, 6]))

    def test_initialize_with_overlapping_stream_params(self):
        fn = lambda xy0, xy1: Points([xy0['x'], xy0['y']]) + Curve([xy1['x'], xy1['y']])
        xy_stream0 = XY(x=1, y=2)
        xy_stream1 = XY(x=3, y=4)
        dmap = DynamicMap(fn, streams=[xy_stream0, xy_stream1], positional_stream_args=True)
        self.assertEqual(dmap[()], Points([1, 2]) + Curve([3, 4]))