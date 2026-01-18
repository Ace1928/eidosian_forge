import numpy as np
import pyviz_comms as comms
from bokeh.models import (
from param import concrete_descendents
from holoviews import Curve
from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import Callback
from holoviews.plotting.bokeh.element import ElementPlot
from .. import option_intersections
def _test_colormapping(self, element, dim, log=False):
    plot = bokeh_renderer.get_plot(element)
    plot.initialize_plot()
    cmapper = plot.handles['color_mapper']
    low, high = element.range(dim)
    self.assertEqual(cmapper.low, low)
    self.assertEqual(cmapper.high, high)
    mapper_type = LogColorMapper if log else LinearColorMapper
    self.assertTrue(isinstance(cmapper, mapper_type))