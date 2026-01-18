import numpy as np
import param
from ...element import HLine, HSpan, Tiles, VLine, VSpan
from ..mixins import GeomMixin
from .element import ElementPlot
class ShapePlot(ElementPlot):
    _shape_type = None
    style_opts = ['opacity', 'fillcolor', 'line_color', 'line_width', 'line_dash']
    _supports_geo = True

    def init_graph(self, datum, options, index=0, is_geo=False, **kwargs):
        if is_geo:
            trace = {'type': 'scattermapbox', 'mode': 'lines', 'showlegend': False, 'hoverinfo': 'skip'}
            trace.update(datum, **options)
            if options.get('fillcolor', None):
                trace['fill'] = 'toself'
            return dict(traces=[trace])
        else:
            shape = dict(type=self._shape_type, **dict(datum, **options))
            return dict(shapes=[shape])

    @staticmethod
    def build_path(xs, ys, closed=True):
        line_tos = ''.join([f'L{x} {y}' for x, y in zip(xs[1:], ys[1:])])
        path = f'M{xs[0]} {ys[0]}{line_tos}'
        if closed:
            path += 'Z'
        return path