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
class _TestSubscriber:

    def __init__(self, cb=None):
        self.call_count = 0
        self.kwargs = None
        self.cb = cb

    def __call__(self, **kwargs):
        self.call_count += 1
        self.kwargs = kwargs
        if self.cb:
            self.cb()