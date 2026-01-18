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
class TextPlot(ElementPlot, AnnotationPlot):
    style_opts = text_properties + ['color', 'angle', 'visible']
    _plot_methods = dict(single='text', batched='text')
    selection_display = None

    def get_data(self, element, ranges, style):
        mapping = dict(x='x', y='y', text='text')
        if self.static_source:
            return (dict(x=[], y=[], text=[]), mapping, style)
        if self.invert_axes:
            data = dict(x=[element.y], y=[element.x])
        else:
            data = dict(x=[element.x], y=[element.y])
        self._categorize_data(data, ('x', 'y'), element.dimensions())
        data['text'] = [element.text]
        if 'text_align' not in style:
            style['text_align'] = element.halign
        baseline = 'middle' if element.valign == 'center' else element.valign
        if 'text_baseline' not in style:
            style['text_baseline'] = baseline
        if 'text_font_size' not in style:
            style['text_font_size'] = '%dPt' % element.fontsize
        if 'color' in style:
            style['text_color'] = style.pop('color')
        style['angle'] = np.deg2rad(style.get('angle', element.rotation))
        return (data, mapping, style)

    def get_batched_data(self, element, ranges=None):
        data = defaultdict(list)
        zorders = self._updated_zorders(element)
        for (_key, el), zorder in zip(element.data.items(), zorders):
            style = self.lookup_options(element.last, 'style')
            style = style.max_cycles(len(self.ordering))[zorder]
            eldata, elmapping, style = self.get_data(el, ranges, style)
            for k, eld in eldata.items():
                data[k].extend(eld)
        return (data, elmapping, style)

    def get_extents(self, element, ranges=None, range_type='combined', **kwargs):
        return (None, None, None, None)