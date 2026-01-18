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
class _SyntheticAnnotationPlot(ColorbarPlot):
    apply_ranges = param.Boolean(default=True, doc='\n        Whether to include the annotation in axis range calculations.')
    style_opts = [*line_properties, 'level', 'visible']
    _allow_implicit_categories = False

    def __init__(self, element, **kwargs):
        if not bokeh32:
            name = type(getattr(element, 'last', element)).__name__
            msg = f'{name} element requires Bokeh >=3.2'
            raise ImportError(msg)
        super().__init__(element, **kwargs)

    def _init_glyph(self, plot, mapping, properties):
        self._plot_methods = {'single': self._methods[self.invert_axes]}
        return super()._init_glyph(plot, mapping, properties)

    def get_data(self, element, ranges, style):
        data = element.columns(element.kdims)
        self._get_hover_data(data, element)
        default = self._element_default[self.invert_axes].kdims
        mapping = {str(d): str(k) for d, k in zip(default, element.kdims)}
        return (data, mapping, style)

    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        figure = super().initialize_plot(ranges=ranges, plot=plot, plots=plots, source=source)
        if self.overlaid and set(itertools.chain.from_iterable(ranges)) - {'HSpans', 'VSpans', 'VLines', 'HLines'}:
            return figure
        labels = [self.xlabel or 'x', self.ylabel or 'y']
        labels = labels[::-1] if self.invert_axes else labels
        for ax, label in zip(figure.axis, labels):
            ax.axis_label = label
        return figure

    def get_extents(self, element, ranges=None, range_type='combined', **kwargs):
        extents = super().get_extents(element, ranges, range_type)
        if isinstance(element, HLines):
            extents = (np.nan, extents[0], np.nan, extents[2])
        elif isinstance(element, VLines):
            extents = (extents[0], np.nan, extents[2], np.nan)
        elif isinstance(element, HSpans):
            extents = pd.array(extents)
            extents = (np.nan, extents[:2].min(), np.nan, extents[2:].max())
        elif isinstance(element, VSpans):
            extents = pd.array(extents)
            extents = (extents[:2].min(), np.nan, extents[2:].max(), np.nan)
        return extents