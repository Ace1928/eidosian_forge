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
class IconMixin(Widget):
    icon = param.String(default=None, doc='\n        An icon to render to the left of the button label. Either an SVG or an\n        icon name which is loaded from https://tabler-icons.io.')
    icon_size = param.String(default='1em', doc='\n        Size of the icon as a string, e.g. 12px or 1em.')
    _rename: ClassVar[Mapping[str, str | None]] = {'icon_size': None, '_icon': 'icon', 'icon': None}
    __abstract = True

    def __init__(self, **params) -> None:
        self._rename = dict(self._rename, **IconMixin._rename)
        super().__init__(**params)

    def _process_param_change(self, params):
        icon_size = params.pop('icon_size', self.icon_size)
        if params.get('icon') is not None:
            icon = params['icon']
            if icon.lstrip().startswith('<svg'):
                icon_model = SVGIcon(svg=icon, size=icon_size)
            else:
                icon_model = TablerIcon(icon_name=icon, size=icon_size)
            params['_icon'] = icon_model
        return super()._process_param_change(params)