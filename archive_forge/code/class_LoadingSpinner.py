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
class LoadingSpinner(BooleanIndicator):
    """
    The `LoadingSpinner` is a boolean indicator providing a visual
    representation of the loading status.

    If the value is set to `True` the spinner will rotate while
    setting it to `False` will disable the rotating segment.

    Reference: https://panel.holoviz.org/reference/indicators/LoadingSpinner.html

    :Example:

    >>> LoadingSpinner(value=True, color='primary', bgcolor='light', width=100, height=100)
    """
    bgcolor = param.ObjectSelector(default='light', objects=['dark', 'light'])
    color = param.ObjectSelector(default='dark', objects=['primary', 'secondary', 'success', 'info', 'danger', 'warning', 'light', 'dark'])
    size = param.Integer(default=125, doc='\n        Size of the spinner in pixels.')
    value = param.Boolean(default=False, doc='\n        Whether the indicator is active or not.')
    _rename = {'name': 'text'}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'value': None, 'color': None, 'bgcolor': None, 'size': None}
    _stylesheets: ClassVar[List[str]] = [f'{CDN_DIST}css/loadingspinner.css']
    _widget_type: ClassVar[Type[Model]] = HTML

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        if 'text' in msg:
            text = msg.pop('text')
            if not PARAM_NAME_PATTERN.match(text):
                msg['text'] = escape(f'<span><b>{text}</b></span>')
        value = msg.pop('value', None)
        color = msg.pop('color', None)
        bgcolor = msg.pop('bgcolor', None)
        if msg.get('sizing_mode') == 'fixed':
            msg['sizing_mode'] = None
        if 'size' in msg or 'height' in msg or 'stylesheets' in msg:
            if 'width' in msg and msg['width'] == msg.get('height'):
                del msg['width']
            size = int(min(msg.pop('height', self.height) or float('inf'), msg.pop('size', self.size)))
            msg['stylesheets'] = [f':host {{ --loading-spinner-size: {size}px; }}'] + msg.get('stylesheets', [])
            msg['min_width'] = msg['min_height'] = size
        if value is None and (not (color or bgcolor)):
            return msg
        color_cls = f'{self.color}-{self.bgcolor}'
        msg['css_classes'] = ['loader', 'spin', color_cls] if self.value else ['loader', self.bgcolor]
        return msg