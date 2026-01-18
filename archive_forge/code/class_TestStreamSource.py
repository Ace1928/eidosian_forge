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
class TestStreamSource(ComparisonTestCase):

    def tearDown(self):
        with param.logging_level('ERROR'):
            Stream.registry = defaultdict(list)

    def test_source_empty_element(self):
        points = Points([])
        stream = PointerX(source=points)
        self.assertIs(stream.source, points)

    def test_source_empty_element_remap(self):
        points = Points([])
        stream = PointerX(source=points)
        self.assertIs(stream.source, points)
        curve = Curve([])
        stream.source = curve
        self.assertNotIn(points, Stream.registry)
        self.assertIn(curve, Stream.registry)

    def test_source_empty_dmap(self):
        points_dmap = DynamicMap(lambda x: Points([]), kdims=['X'])
        stream = PointerX(source=points_dmap)
        self.assertIs(stream.source, points_dmap)

    def test_source_registry(self):
        points = Points([(0, 0)])
        PointerX(source=points)
        self.assertIn(points, Stream.registry)

    def test_source_registry_empty_element(self):
        points = Points([])
        PointerX(source=points)
        self.assertIn(points, Stream.registry)