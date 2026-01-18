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
class DivPlot(BokehPlot, GenericElementPlot, AnnotationPlot):
    height = param.Number(default=300)
    width = param.Number(default=300)
    sizing_mode = param.ObjectSelector(default=None, objects=['fixed', 'stretch_width', 'stretch_height', 'stretch_both', 'scale_width', 'scale_height', 'scale_both', None], doc='\n\n        How the component should size itself.\n\n        * "fixed" :\n          Component is not responsive. It will retain its original\n          width and height regardless of any subsequent browser window\n          resize events.\n        * "stretch_width"\n          Component will responsively resize to stretch to the\n          available width, without maintaining any aspect ratio. The\n          height of the component depends on the type of the component\n          and may be fixed or fit to component\'s contents.\n        * "stretch_height"\n          Component will responsively resize to stretch to the\n          available height, without maintaining any aspect ratio. The\n          width of the component depends on the type of the component\n          and may be fixed or fit to component\'s contents.\n        * "stretch_both"\n          Component is completely responsive, independently in width\n          and height, and will occupy all the available horizontal and\n          vertical space, even if this changes the aspect ratio of the\n          component.\n        * "scale_width"\n          Component will responsively resize to stretch to the\n          available width, while maintaining the original or provided\n          aspect ratio.\n        * "scale_height"\n          Component will responsively resize to stretch to the\n          available height, while maintaining the original or provided\n          aspect ratio.\n        * "scale_both"\n          Component will responsively resize to both the available\n          width and height, while maintaining the original or provided\n          aspect ratio.\n    ')
    hooks = param.HookList(default=[], doc='\n        Optional list of hooks called when finalizing a plot. The\n        hook is passed the plot object and the displayed element, and\n        other plotting handles can be accessed via plot.handles.')
    _stream_data = False
    selection_display = None

    def __init__(self, element, plot=None, **params):
        super().__init__(element, **params)
        self.callbacks = []
        self.handles = {} if plot is None else self.handles['plot']
        self.static = len(self.hmap) == 1 and len(self.keys) == len(self.hmap)

    def get_data(self, element, ranges, style):
        return (element.data, {}, style)

    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        """
        Initializes a new plot object with the last available frame.
        """
        element = self.hmap.last
        key = self.keys[-1]
        self.current_frame = element
        self.current_key = key
        data, _, _ = self.get_data(element, ranges, {})
        div = HTML(text=escape(data), width=self.width, height=self.height, sizing_mode=self.sizing_mode)
        self.handles['plot'] = div
        self._execute_hooks(element)
        self.drawn = True
        return div

    def update_frame(self, key, ranges=None, plot=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        element = self._get_frame(key)
        text, _, _ = self.get_data(element, ranges, {})
        self.state.update(text=text, sizing_mode=self.sizing_mode)