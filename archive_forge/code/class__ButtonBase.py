from __future__ import annotations
from typing import (
import param
from bokeh.events import ButtonClick, MenuItemClick
from bokeh.models import Dropdown as _BkDropdown, Toggle as _BkToggle
from bokeh.models.ui import SVGIcon, TablerIcon
from ..io.resources import CDN_DIST
from ..links import Callback
from ..models.widgets import Button as _BkButton
from ._mixin import TooltipMixin
from .base import Widget
class _ButtonBase(Widget):
    button_type = param.ObjectSelector(default='default', objects=BUTTON_TYPES, doc="\n        A button theme; should be one of 'default' (white), 'primary'\n        (blue), 'success' (green), 'info' (yellow), 'light' (light),\n        or 'danger' (red).")
    button_style = param.ObjectSelector(default='solid', objects=BUTTON_STYLES, doc="\n        A button style to switch between 'solid', 'outline'.")
    _rename: ClassVar[Mapping[str, str | None]] = {'name': 'label', 'button_style': None}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'button_style': None}
    _stylesheets: ClassVar[List[str]] = [f'{CDN_DIST}css/button.css']
    __abstract = True

    def _process_param_change(self, params):
        if 'button_style' in params or 'css_classes' in params:
            params['css_classes'] = [params.pop('button_style', self.button_style)] + params.get('css_classes', self.css_classes)
        return super()._process_param_change(params)