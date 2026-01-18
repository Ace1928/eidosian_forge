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
class TestDerivedStream(ComparisonTestCase):

    def test_simple_derived_stream(self):
        v0 = Val(v=1.0)
        v1 = Val(v=2.0)
        s0 = Sum([v0, v1])
        self.assertEqual(s0.v, 3.0)
        v0.event(v=7.0)
        self.assertEqual(s0.v, 9.0)
        v1.event(v=-8.0)
        self.assertEqual(s0.v, -1.0)

    def test_nested_derived_stream(self):
        v0 = Val(v=1.0)
        v1 = Val(v=4.0)
        v2 = Val(v=7.0)
        s1 = Sum([v0, v1])
        s0 = Sum([s1, v2])
        self.assertEqual(s0.v, 12.0)
        v2.event(v=8.0)
        self.assertEqual(s0.v, 13.0)
        v1.event(v=5.0)
        self.assertEqual(s0.v, 14.0)

    def test_derived_stream_constants(self):
        v0 = Val(v=1.0)
        v1 = Val(v=4.0)
        v2 = Val(v=7.0)
        s0 = Sum([v0, v1, v2], base=100)
        self.assertEqual(s0.v, 112.0)
        v2.event(v=8.0)
        self.assertEqual(s0.v, 113.0)

    def test_exclusive_derived_stream(self):
        v0 = Val()
        v1 = Val(v=2.0)
        s0 = Sum([v0, v1], exclusive=True)
        self.assertEqual(s0.v, 2.0)
        v0.event(v=7.0)
        self.assertEqual(s0.v, 7.0)
        v1.event(v=-8.0)
        self.assertEqual(s0.v, -8.0)