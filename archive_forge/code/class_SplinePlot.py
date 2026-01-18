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
class SplinePlot(ElementPlot, AnnotationPlot):
    """
    Draw the supplied Spline annotation (see Spline docstring).
    Does not support matplotlib Path codes.
    """
    style_opts = line_properties + ['visible']
    _plot_methods = dict(single='bezier')
    selection_display = None

    def get_data(self, element, ranges, style):
        if self.invert_axes:
            data_attrs = ['y0', 'x0', 'cy0', 'cx0', 'cy1', 'cx1', 'y1', 'x1']
        else:
            data_attrs = ['x0', 'y0', 'cx0', 'cy0', 'cx1', 'cy1', 'x1', 'y1']
        verts = np.array(element.data[0])
        inds = np.where(np.array(element.data[1]) == 1)[0]
        data = {da: [] for da in data_attrs}
        skipped = False
        for vs in np.split(verts, inds[1:]):
            if len(vs) != 4:
                skipped = len(vs) > 1
                continue
            for x, y, xl, yl in zip(vs[:, 0], vs[:, 1], data_attrs[::2], data_attrs[1::2]):
                data[xl].append(x)
                data[yl].append(y)
        if skipped:
            self.param.warning('Bokeh SplinePlot only support cubic splines, unsupported splines were skipped during plotting.')
        data = {da: data[da] for da in data_attrs}
        return (data, dict(zip(data_attrs, data_attrs)), style)