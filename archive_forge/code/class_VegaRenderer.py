import warnings
import json
import random
from .base import Renderer
from ..exporter import Exporter
class VegaRenderer(Renderer):

    def open_figure(self, fig, props):
        self.props = props
        self.figwidth = int(props['figwidth'] * props['dpi'])
        self.figheight = int(props['figheight'] * props['dpi'])
        self.data = []
        self.scales = []
        self.axes = []
        self.marks = []

    def open_axes(self, ax, props):
        if len(self.axes) > 0:
            warnings.warn('multiple axes not yet supported')
        self.axes = [dict(type='x', scale='x', ticks=10), dict(type='y', scale='y', ticks=10)]
        self.scales = [dict(name='x', domain=props['xlim'], type='linear', range='width'), dict(name='y', domain=props['ylim'], type='linear', range='height')]

    def draw_line(self, data, coordinates, style, label, mplobj=None):
        if coordinates != 'data':
            warnings.warn('Only data coordinates supported. Skipping this')
        dataname = 'table{0:03d}'.format(len(self.data) + 1)
        self.data.append({'name': dataname, 'values': [dict(x=d[0], y=d[1]) for d in data]})
        self.marks.append({'type': 'line', 'from': {'data': dataname}, 'properties': {'enter': {'interpolate': {'value': 'monotone'}, 'x': {'scale': 'x', 'field': 'data.x'}, 'y': {'scale': 'y', 'field': 'data.y'}, 'stroke': {'value': style['color']}, 'strokeOpacity': {'value': style['alpha']}, 'strokeWidth': {'value': style['linewidth']}}}})

    def draw_markers(self, data, coordinates, style, label, mplobj=None):
        if coordinates != 'data':
            warnings.warn('Only data coordinates supported. Skipping this')
        dataname = 'table{0:03d}'.format(len(self.data) + 1)
        self.data.append({'name': dataname, 'values': [dict(x=d[0], y=d[1]) for d in data]})
        self.marks.append({'type': 'symbol', 'from': {'data': dataname}, 'properties': {'enter': {'interpolate': {'value': 'monotone'}, 'x': {'scale': 'x', 'field': 'data.x'}, 'y': {'scale': 'y', 'field': 'data.y'}, 'fill': {'value': style['facecolor']}, 'fillOpacity': {'value': style['alpha']}, 'stroke': {'value': style['edgecolor']}, 'strokeOpacity': {'value': style['alpha']}, 'strokeWidth': {'value': style['edgewidth']}}}})

    def draw_text(self, text, position, coordinates, style, text_type=None, mplobj=None):
        if text_type == 'xlabel':
            self.axes[0]['title'] = text
        elif text_type == 'ylabel':
            self.axes[1]['title'] = text