from __future__ import annotations
import datetime as dt
from typing import (
import numpy as np
import param
from bokeh.models import CustomJS
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from bokeh.models.widgets.sliders import NumericalSlider as _BkNumericalSlider
from param.parameterized import resolve_value
from ..config import config
from ..io import state
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..util import (
from ..viewable import Layoutable
from ..widgets import FloatInput, IntInput
from .base import CompositeWidget, Widget
from .input import StaticText
class _EditableContinuousSlider(CompositeWidget):
    """
    The EditableFloatSlider extends the FloatSlider by adding a text
    input field to manually edit the value and potentially override
    the bounds.
    """
    editable = param.Boolean(default=True, doc='\n        Whether the value is editable via the text input.')
    show_value = param.Boolean(default=False, readonly=True, precedence=-1, doc='\n        Whether to show the widget value.')
    _composite_type: ClassVar[Type[Panel]] = Column
    _slider_widget: ClassVar[Type[Widget]]
    _input_widget: ClassVar[Type[Widget]]
    __abstract = True

    def __init__(self, **params):
        if 'width' not in params and 'sizing_mode' not in params:
            params['width'] = 300
        self._validate_init_bounds(params)
        super().__init__(**params)
        self._label = StaticText(margin=0, align='end')
        self._slider = self._slider_widget(value=self.value, margin=(0, 0, 5, 0), sizing_mode='stretch_width', tags=['composite'])
        self._slider.param.watch(self._sync_value, 'value')
        self._slider.param.watch(self._sync_value, 'value_throttled')
        self._value_edit = self._input_widget(margin=0, align='end', css_classes=['slider-edit'], stylesheets=[f'{CDN_DIST}css/editable_slider.css'], format=self.format)
        self._value_edit.param.watch(self._sync_value, 'value')
        self._value_edit.param.watch(self._sync_value, 'value_throttled')
        self._value_edit.jscallback(args={'slider': self._slider}, value='\n        if (cb_obj.value < slider.start)\n          slider.start = cb_obj.value\n        else if (cb_obj.value > slider.end)\n          slider.end = cb_obj.value\n        ')
        label = Row(self._label, self._value_edit)
        self._composite.extend([label, self._slider])
        self._update_disabled()
        self._update_editable()
        self._update_layout()
        self._update_name()
        self._update_slider()
        self._update_value()
        self._update_bounds()

    def _validate_init_bounds(self, params):
        """
        This updates the default value, start and end
        if outside the fixed_start and fixed_end
        """
        start, end = (None, None)
        if 'start' not in params:
            if 'fixed_start' in params:
                start = params['fixed_start']
            elif 'end' in params:
                start = params.get('end') - params.get('step', 1)
            elif 'fixed_end' in params:
                start = params.get('fixed_end') - params.get('step', 1)
        if 'end' not in params:
            if 'fixed_end' in params:
                end = params['fixed_end']
            elif 'start' in params:
                end = params['start'] + params.get('step', 1)
            elif 'fixed_start' in params:
                end = params['fixed_start'] + params.get('step', 1)
        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end
        if 'value' not in params and 'start' in params:
            params['value'] = params['start']
        if 'value' not in params and 'end' in params:
            params['value'] = params['end']

    @param.depends('width', 'height', 'sizing_mode', watch=True)
    def _update_layout(self):
        self._value_edit.sizing_mode = self.sizing_mode
        if self.sizing_mode not in ('stretch_width', 'stretch_both'):
            w = (self.width or 300) // 4
            self._value_edit.width = w

    @param.depends('disabled', 'editable', watch=True)
    def _update_editable(self):
        self._value_edit.disabled = not self.editable or self.disabled

    @param.depends('disabled', watch=True)
    def _update_disabled(self):
        self._slider.disabled = self.disabled

    @param.depends('name', watch=True)
    def _update_name(self):
        if self.name:
            label = f'{self.name}:'
            margin = (0, 10, 0, 0)
        else:
            label = ''
            margin = (0, 0, 0, 0)
        self._label.param.update(margin=margin, value=label)

    @param.depends('start', 'end', 'step', 'bar_color', 'direction', 'show_value', 'tooltips', 'format', watch=True)
    def _update_slider(self):
        self._slider.param.update(format=self.format, start=self.start, end=self.end, step=self.step, bar_color=self.bar_color, direction=self.direction, show_value=self.show_value, tooltips=self.tooltips)
        self._value_edit.step = self.step

    @param.depends('value', watch=True)
    def _update_value(self):
        self._slider.value = self.value
        self._value_edit.value = self.value

    def _sync_value(self, event):
        with param.edit_constant(self):
            self.param.update(**{event.name: event.new})

    @param.depends('start', 'end', 'fixed_start', 'fixed_end', watch=True)
    def _update_bounds(self):
        self.param.value.softbounds = (self.start, self.end)
        self.param.value_throttled.softbounds = (self.start, self.end)
        self.param.value.bounds = (self.fixed_start, self.fixed_end)
        self.param.value_throttled.bounds = (self.fixed_start, self.fixed_end)
        if self.fixed_start is not None:
            self._slider.start = max(self.fixed_start, self.start)
        if self.fixed_end is not None:
            self._slider.end = min(self.fixed_end, self.end)
        self._value_edit.start = self.fixed_start
        self._value_edit.end = self.fixed_end