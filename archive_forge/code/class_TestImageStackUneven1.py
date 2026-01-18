from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
class TestImageStackUneven1(_ImageStackBase):
    __test__ = True

    def setUp(self):
        self.a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3, [np.nan] * 3])
        self.b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3, [np.nan] * 3])
        self.c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1], [np.nan] * 3])
        self.ysize = 4
        self.xsize = 3
        super().setUp()