import itertools
from collections import defaultdict
from html import escape
import numpy as np
import pandas as pd
import param
from bokeh.models import Arrow, BoxAnnotation, NormalHead, Slope, Span, TeeHead
from bokeh.transform import dodge
from panel.models import HTML
from ...core.util import datetime_types, dimension_sanitizer
from ...element import HLine, HLines, HSpans, VLine, VLines, VSpan, VSpans
from ..plot import GenericElementPlot
from .element import AnnotationPlot, ColorbarPlot, CompositeElementPlot, ElementPlot
from .plot import BokehPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
from .util import bokeh32, date_to_integer
class SlopePlot(ElementPlot, AnnotationPlot):
    style_opts = line_properties + ['level']
    _plot_methods = dict(single='Slope')
    selection_display = None

    def get_data(self, element, ranges, style):
        data, mapping = ({}, {})
        gradient, intercept = element.data
        if self.invert_axes:
            if gradient == 0:
                gradient = (np.inf, np.inf)
            else:
                gradient, intercept = (1 / gradient, -(intercept / gradient))
        mapping['gradient'] = gradient
        mapping['y_intercept'] = intercept
        return (data, mapping, style)

    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        slope = Slope(level=properties.get('level', 'glyph'), **mapping)
        plot.add_layout(slope)
        return (None, slope)

    def get_extents(self, element, ranges=None, range_type='combined', **kwargs):
        return (None, None, None, None)