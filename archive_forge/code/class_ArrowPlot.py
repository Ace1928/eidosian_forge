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
class ArrowPlot(CompositeElementPlot, AnnotationPlot):
    style_opts = [f'arrow_{p}' for p in line_properties + fill_properties + ['size']] + text_properties
    _style_groups = {'arrow': 'arrow', 'text': 'text'}
    _draw_order = ['arrow_1', 'text_1']
    selection_display = None

    def get_data(self, element, ranges, style):
        plot = self.state
        label_mapping = dict(x='x', y='y', text='text')
        arrow_mapping = dict(x_start='x_start', x_end='x_end', y_start='y_start', y_end='y_end')
        x1, y1 = (element.x, element.y)
        axrange = plot.x_range if self.invert_axes else plot.y_range
        span = (axrange.end - axrange.start) / 6.0
        if element.direction == '^':
            x2, y2 = (x1, y1 - span)
            label_mapping['text_baseline'] = 'top'
        elif element.direction == '<':
            x2, y2 = (x1 + span, y1)
            label_mapping['text_align'] = 'left'
            label_mapping['text_baseline'] = 'middle'
        elif element.direction == '>':
            x2, y2 = (x1 - span, y1)
            label_mapping['text_align'] = 'right'
            label_mapping['text_baseline'] = 'middle'
        else:
            x2, y2 = (x1, y1 + span)
            label_mapping['text_baseline'] = 'bottom'
        arrow_data = {'x_end': [x1], 'y_end': [y1], 'x_start': [x2], 'y_start': [y2]}
        arrow_mapping['arrow_start'] = arrow_start.get(element.arrowstyle, None)
        arrow_mapping['arrow_end'] = arrow_end.get(element.arrowstyle, NormalHead)
        if self.invert_axes:
            label_data = dict(x=[y2], y=[x2])
        else:
            label_data = dict(x=[x2], y=[y2])
        label_data['text'] = [element.text]
        return ({'text_1': label_data, 'arrow_1': arrow_data}, {'arrow_1': arrow_mapping, 'text_1': label_mapping}, style)

    def _init_glyph(self, plot, mapping, properties, key):
        """
        Returns a Bokeh glyph object.
        """
        properties = {k: v for k, v in properties.items() if 'legend' not in k}
        if key == 'arrow_1':
            source = properties.pop('source')
            arrow_end = mapping.pop('arrow_end')
            arrow_start = mapping.pop('arrow_start')
            for p in ('alpha', 'color'):
                v = properties.pop(p, None)
                for t in ('line', 'fill'):
                    if v is None:
                        continue
                    key = f'{t}_{p}'
                    if key not in properties:
                        properties[key] = v
            start = arrow_start(**properties) if arrow_start else None
            end = arrow_end(**properties) if arrow_end else None
            line_props = {p: v for p, v in properties.items() if p.startswith('line_')}
            renderer = Arrow(start=start, end=end, source=source, **dict(line_props, **mapping))
            glyph = renderer
        else:
            properties = {p if p == 'source' else 'text_' + p: v for p, v in properties.items()}
            renderer, glyph = super()._init_glyph(plot, mapping, properties, key)
        plot.renderers.append(renderer)
        return (renderer, glyph)

    def get_extents(self, element, ranges=None, range_type='combined', **kwargs):
        return (None, None, None, None)