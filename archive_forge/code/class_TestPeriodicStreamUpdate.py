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
class TestPeriodicStreamUpdate(ComparisonTestCase):

    def test_periodic_counter_blocking(self):

        class Counter:

            def __init__(self):
                self.count = 0

            def __call__(self):
                self.count += 1
                return Curve([1, 2, 3])
        next_stream = Stream.define('Next')()
        counter = Counter()
        dmap = DynamicMap(counter, streams=[next_stream])
        next_stream.add_subscriber(lambda **kwargs: dmap[()])
        dmap.periodic(0.01, 100)
        self.assertEqual(counter.count, 100)

    def test_periodic_param_fn_blocking(self):

        def callback(x):
            return Curve([1, 2, 3])
        xval = Stream.define('x', x=0)()
        dmap = DynamicMap(callback, streams=[xval])
        xval.add_subscriber(lambda **kwargs: dmap[()])
        dmap.periodic(0.01, 100, param_fn=lambda i: {'x': i})
        self.assertEqual(xval.x, 100)

    @pytest.mark.flaky(reruns=3)
    def test_periodic_param_fn_non_blocking(self):

        def callback(x):
            return Curve([1, 2, 3])
        xval = Stream.define('x', x=0)()
        dmap = DynamicMap(callback, streams=[xval])
        xval.add_subscriber(lambda **kwargs: dmap[()])
        self.assertNotEqual(xval.x, 100)
        dmap.periodic(0.0001, 100, param_fn=lambda i: {'x': i}, block=False)
        time.sleep(2)
        if not dmap.periodic.instance.completed:
            raise RuntimeError('Periodic callback timed out.')
        dmap.periodic.stop()
        self.assertEqual(xval.x, 100)

    def test_periodic_param_fn_blocking_period(self):

        def callback(x):
            return Curve([1, 2, 3])
        xval = Stream.define('x', x=0)()
        dmap = DynamicMap(callback, streams=[xval])
        xval.add_subscriber(lambda **kwargs: dmap[()])
        start = time.time()
        dmap.periodic(0.5, 10, param_fn=lambda i: {'x': i}, block=True)
        end = time.time()
        self.assertEqual(end - start > 5, True)

    def test_periodic_param_fn_blocking_timeout(self):

        def callback(x):
            return Curve([1, 2, 3])
        xval = Stream.define('x', x=0)()
        dmap = DynamicMap(callback, streams=[xval])
        xval.add_subscriber(lambda **kwargs: dmap[()])
        start = time.time()
        dmap.periodic(0.5, 100, param_fn=lambda i: {'x': i}, timeout=3)
        end = time.time()
        self.assertEqual(end - start < 5, True)