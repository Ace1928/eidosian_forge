from __future__ import annotations
import asyncio
import math
import os
import sys
import time
from math import pi
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource, FixedTicker, Tooltip
from bokeh.plotting import figure
from tqdm.asyncio import tqdm as _tqdm
from .._param import Align
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..models import (
from ..pane.markup import Str
from ..reactive import SyncableData
from ..util import PARAM_NAME_PATTERN, escape, updating
from ..viewable import Viewable
from .base import Widget
class LinearGauge(ValueIndicator):
    """
    A LinearGauge represents a value in some range as a position on an
    linear scale. It is similar to a Dial/Gauge but visually more
    compact.

    Reference: https://panel.holoviz.org/reference/indicators/LinearGauge.html

    :Example:

    >>> LinearGauge(value=30, default_color='red', bounds=(0, 100))
    """
    bounds = param.Range(default=(0, 100), doc='\n      The upper and lower bound of the gauge.')
    default_color = param.String(default='lightblue', doc='\n      Color of the radial annulus if not color thresholds are supplied.')
    colors = param.Parameter(default=None, doc='\n      Color thresholds for the gauge, specified as a list of tuples\n      of the fractional threshold and the color to switch to.')
    format = param.String(default='{value:.2f}%', doc='\n      Formatting string for the value indicator and lower/upper bounds.')
    height = param.Integer(default=300, bounds=(1, None))
    horizontal = param.Boolean(default=False, doc='\n      Whether to display the linear gauge horizontally.')
    nan_format = param.String(default='-', doc='\n      How to format nan values.')
    needle_color = param.String(default='black', doc='\n      Color of the gauge needle.')
    show_boundaries = param.Boolean(default=False, doc='\n      Whether to show the boundaries between colored regions.')
    unfilled_color = param.String(default='whitesmoke', doc='\n      Color of the unfilled region of the LinearGauge.')
    title_size = param.String(default=None, doc='\n      Font size of the gauge title.')
    tick_size = param.String(default=None, doc='\n      Font size of the gauge tick labels.')
    value_size = param.String(default=None, doc='\n      Font size of the gauge value label.')
    value = param.Number(default=25, allow_None=True, doc='\n      Value to indicate on the dial a value within the declared bounds.')
    width = param.Integer(default=125, bounds=(1, None))
    _manual_params = ['value', 'bounds', 'format', 'title_size', 'value_size', 'horizontal', 'height', 'colors', 'tick_size', 'unfilled_color', 'width', 'nan_format', 'needle_color']
    _data_params = ['value', 'bounds', 'format', 'nan_format', 'needle_color', 'colors']
    _rerender_params = ['horizontal']
    _rename: ClassVar[Mapping[str, str | None]] = {'background': 'background_fill_color', 'name': 'name', 'show_boundaries': None, 'default_color': None}
    _updates = False

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value_bounds()

    @param.depends('bounds', watch=True)
    def _update_value_bounds(self):
        self.param.value.bounds = self.bounds

    @property
    def _color_intervals(self):
        vmin, vmax = self.bounds
        value = self.value
        ncolors = len(self.colors) if self.colors else 1
        interval = vmax - vmin
        if math.isfinite(value):
            fraction = value / interval
            idx = round(fraction * (ncolors - 1))
        else:
            fraction = 0
            idx = 0
        if not self.colors:
            intervals = [(fraction, self.default_color)]
            intervals.append((1, self.unfilled_color))
        elif self.show_boundaries:
            intervals = [c if isinstance(c, tuple) else ((i + 1) / ncolors, c) for i, c in enumerate(self.colors)]
        else:
            intervals = [self.colors[idx] if isinstance(self.colors[0], tuple) else (fraction, self.colors[idx])]
            intervals.append((1, self.unfilled_color))
        return intervals

    def _get_data(self, properties):
        vmin, vmax = self.bounds
        value = self.value
        interval = vmax - vmin
        colors, values = ([], [vmin])
        above = False
        prev = None
        for v, color in self._color_intervals:
            val = v * interval
            if val == prev:
                continue
            elif val > value:
                if not above:
                    colors.append(color)
                    values.append(value)
                    above = True
                color = self.unfilled_color
            colors.append(color)
            values.append(val)
            prev = val
        value = self.format.format(value=value).replace('nan', self.nan_format)
        return ({'y0': values[:-1], 'y1': values[1:], 'color': colors}, {'y': [self.value], 'text': [value]})

    def _get_model(self, doc, root=None, parent=None, comm=None):
        params = self._get_properties(doc)
        model = figure(outline_line_color=None, toolbar_location=None, tools=[], x_axis_location='above', y_axis_location='right', **params)
        model.grid.visible = False
        model.xaxis.major_label_standoff = 2
        model.yaxis.major_label_standoff = 2
        model.xaxis.axis_label_standoff = 2
        model.yaxis.axis_label_standoff = 2
        self._update_name(model)
        self._update_title_size(model)
        self._update_tick_size(model)
        self._update_figure(model)
        self._update_axes(model)
        self._update_renderers(model)
        self._update_bounds(model)
        self._design.apply_bokeh_theme_to_model(model)
        root = root or model
        self._models[root.ref['id']] = (model, parent)
        return model

    def _update_name(self, model):
        model.xaxis.axis_label = self.name
        model.yaxis.axis_label = self.name

    def _update_title_size(self, model):
        title_size = self.title_size or f'{self.width / 6}px'
        model.xaxis.axis_label_text_font_size = title_size
        model.yaxis.axis_label_text_font_size = title_size

    def _update_tick_size(self, model):
        tick_size = self.tick_size or f'{self.width / 9}px'
        model.xaxis.major_label_text_font_size = tick_size
        model.yaxis.major_label_text_font_size = tick_size

    def _update_renderers(self, model):
        model.renderers = []
        properties = self._get_properties(model.document)
        data, needle_data = self._get_data(properties)
        bar_source = ColumnDataSource(data=data, name='bar_source')
        needle_source = ColumnDataSource(data=needle_data, name='needle_source')
        if self.horizontal:
            model.hbar(y=0.1, left='y0', right='y1', height=1, color='color', source=bar_source)
            wedge_params = {'y': 0.5, 'x': 'y', 'angle': np.deg2rad(180)}
            text_params = {'y': -0.4, 'x': 0, 'text_align': 'left', 'text_baseline': 'top'}
        else:
            model.vbar(x=0.1, bottom='y0', top='y1', width=0.9, color='color', source=bar_source)
            wedge_params = {'x': 0.5, 'y': 'y', 'angle': np.deg2rad(90)}
            text_params = {'x': -0.4, 'y': 0, 'text_align': 'left', 'text_baseline': 'bottom', 'angle': np.deg2rad(90)}
        model.scatter(fill_color=self.needle_color, line_color=self.needle_color, source=needle_source, name='needle_renderer', marker='triangle', size=int(self.width / 8), level='overlay', **wedge_params)
        value_size = self.value_size or f'{self.width / 8}px'
        model.text(text='text', source=needle_source, text_font_size=value_size, **text_params)

    def _update_bounds(self, model):
        if self.horizontal:
            x_range, y_range = (tuple(self.bounds), (-0.8, 0.5))
        else:
            x_range, y_range = ((-0.8, 0.5), tuple(self.bounds))
        model.x_range.update(start=x_range[0], end=x_range[1])
        model.y_range.update(start=y_range[0], end=y_range[1])

    def _update_axes(self, model):
        vmin, vmax = self.bounds
        interval = vmax - vmin
        if self.show_boundaries:
            ticks = [vmin] + [v * interval for v, _ in self._color_intervals]
        else:
            ticks = [vmin, vmax]
        ticker = FixedTicker(ticks=ticks)
        if self.horizontal:
            model.xaxis.visible = True
            model.xaxis.ticker = ticker
            model.yaxis.visible = False
        else:
            model.xaxis.visible = False
            model.yaxis.visible = True
            model.yaxis.ticker = ticker

    def _update_figure(self, model):
        params = self._get_properties(model.document)
        if self.horizontal:
            params.update(width=self.height, height=self.width)
        else:
            params.update(width=self.width, height=self.height)
        model.update(**params)

    def _manual_update(self, events, model, doc, root, parent, comm):
        update_data = False
        for event in events:
            if event.name in ('width', 'height'):
                self._update_figure(model)
            elif event.name == 'bounds':
                self._update_bounds(model)
                self._update_renderers(model)
            elif event.name in self._data_params:
                update_data = True
            elif event.name == 'needle_color':
                needle_r = model.select(name='needle_renderer')
                needle_r.glyph.line_color = event.new
                needle_r.glyph.fill_color = event.new
            elif event.name == 'horizontal':
                self._update_bounds(model)
                self._update_figure(model)
                self._update_axes(model)
                self._update_renderers(model)
            elif event.name == 'name':
                self._update_name(model)
            elif event.name == 'tick_size':
                self._update_tick_size(model)
            elif event.name == 'title_size':
                self._update_title_size(model)
        if not update_data:
            return
        properties = self._get_properties(model.document)
        data, needle_data = self._get_data(properties)
        model.select(name='bar_source').data.update(data)
        model.select(name='needle_source').data.update(needle_data)