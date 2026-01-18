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
class TestSubscribers(ComparisonTestCase):

    def test_exception_subscriber(self):
        subscriber = _TestSubscriber()
        position = PointerXY(subscribers=[subscriber])
        kwargs = dict(x=3, y=4)
        position.event(**kwargs)
        self.assertEqual(subscriber.kwargs, kwargs)

    def test_subscriber_disabled(self):
        subscriber = _TestSubscriber()
        position = PointerXY(subscribers=[subscriber])
        kwargs = dict(x=3, y=4)
        position.update(**kwargs)
        self.assertEqual(subscriber.kwargs, None)

    def test_subscribers(self):
        subscriber1 = _TestSubscriber()
        subscriber2 = _TestSubscriber()
        position = PointerXY(subscribers=[subscriber1, subscriber2])
        kwargs = dict(x=3, y=4)
        position.event(**kwargs)
        self.assertEqual(subscriber1.kwargs, kwargs)
        self.assertEqual(subscriber2.kwargs, kwargs)

    def test_batch_subscriber(self):
        subscriber = _TestSubscriber()
        positionX = PointerX(subscribers=[subscriber])
        positionY = PointerY(subscribers=[subscriber])
        positionX.update(x=5)
        positionY.update(y=10)
        Stream.trigger([positionX, positionY])
        self.assertEqual(subscriber.kwargs, dict(x=5, y=10))
        self.assertEqual(subscriber.call_count, 1)

    def test_batch_subscribers(self):
        subscriber1 = _TestSubscriber()
        subscriber2 = _TestSubscriber()
        positionX = PointerX(subscribers=[subscriber1, subscriber2])
        positionY = PointerY(subscribers=[subscriber1, subscriber2])
        positionX.update(x=50)
        positionY.update(y=100)
        Stream.trigger([positionX, positionY])
        self.assertEqual(subscriber1.kwargs, dict(x=50, y=100))
        self.assertEqual(subscriber1.call_count, 1)
        self.assertEqual(subscriber2.kwargs, dict(x=50, y=100))
        self.assertEqual(subscriber2.call_count, 1)

    def test_pipe_memoization(self):

        def points(data):
            subscriber.call_count += 1
            return Points([(0, data)])
        stream = Pipe(data=0)
        dmap = DynamicMap(points, streams=[stream])

        def cb():
            dmap[()]
        subscriber = _TestSubscriber(cb)
        stream.add_subscriber(subscriber)
        dmap[()]
        stream.send(1)
        self.assertEqual(subscriber.call_count, 3)