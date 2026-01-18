import warnings
from itertools import chain
from types import FunctionType
import bokeh
import bokeh.plotting
import numpy as np
import param
from bokeh.document.events import ModelChangedEvent
from bokeh.models import (
from bokeh.models.axes import CategoricalAxis, DatetimeAxis
from bokeh.models.formatters import (
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.models.mappers import (
from bokeh.models.ranges import DataRange1d, FactorRange, Range1d
from bokeh.models.scales import LogScale
from bokeh.models.tickers import (
from bokeh.models.tools import Tool
from packaging.version import Version
from ...core import CompositeOverlay, Dataset, Dimension, DynamicMap, Element, util
from ...core.options import Keywords, SkipRendering, abbreviated_exception
from ...element import Annotation, Contours, Graph, Path, Tiles, VectorField
from ...streams import Buffer, PlotSize, RangeXY
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_axis_label, dim_range_key, process_cmap
from .plot import BokehPlot
from .styles import (
from .tabular import TablePlot
from .util import (
def _plot_properties(self, key, element):
    """
        Returns a dictionary of plot properties.
        """
    init = 'plot' not in self.handles
    size_multiplier = self.renderer.size / 100.0
    options = self._traverse_options(element, 'plot', ['width', 'height'], defaults=False)
    logger = self.param if init else None
    aspect_props, dimension_props = compute_layout_properties(self.width, self.height, self.frame_width, self.frame_height, options.get('width'), options.get('height'), self.aspect, self.data_aspect, self.responsive, size_multiplier, logger=logger)
    if not init:
        if aspect_props['aspect_ratio'] is None:
            aspect_props['aspect_ratio'] = self.state.aspect_ratio
    plot_props = {'align': self.align, 'margin': self.margin, 'max_width': self.max_width, 'max_height': self.max_height, 'min_width': self.min_width, 'min_height': self.min_height}
    plot_props.update(aspect_props)
    if not self.drawn:
        plot_props.update(dimension_props)
    if self.bgcolor:
        plot_props['background_fill_color'] = self.bgcolor
    if self.border is not None:
        for p in ['left', 'right', 'top', 'bottom']:
            plot_props['min_border_' + p] = self.border
    lod = dict(self.param['lod'].default, **self.lod) if 'lod' in self.param else self.lod
    for lod_prop, v in lod.items():
        plot_props['lod_' + lod_prop] = v
    return plot_props