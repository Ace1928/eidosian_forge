from __future__ import annotations
from typing import (
import param
from ..io.resources import CDN_DIST
from ..models import (
from ._mixin import TooltipMixin
from .base import Widget
from .button import ButtonClick, _ClickButton
class _ClickableIcon(Widget):
    active_icon = param.String(default='', doc='\n        The name of the icon to display when toggled from\n        [tabler-icons.io](https://tabler-icons.io)/ or an SVG.')
    icon = param.String(default='heart', doc='\n        The name of the icon to display from\n        [tabler-icons.io](https://tabler-icons.io)/ or an SVG.')
    size = param.String(default=None, doc="\n        An explicit size specified as a CSS font-size, e.g. '1.5em' or '20px'.")
    value = param.Boolean(default=False, doc='\n        Whether the icon is toggled on or off.')
    _widget_type = _PnClickableIcon
    _rename: ClassVar[Mapping[str, str | None]] = {**TooltipMixin._rename, 'name': 'title'}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'description': None}
    _stylesheets: ClassVar[List[str]] = [f'{CDN_DIST}css/icon.css']

    def __init__(self, **params):
        super().__init__(**params)

    @param.depends('icon', 'active_icon', watch=True, on_init=True)
    def _update_icon(self):
        if not self.icon:
            raise ValueError('The icon parameter must not be empty.')
        icon_is_svg = self.icon.startswith('<svg')
        if icon_is_svg and (not self.active_icon):
            raise ValueError('The active_icon parameter must not be empty if icon is an SVG.')