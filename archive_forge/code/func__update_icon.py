from __future__ import annotations
from typing import (
import param
from ..io.resources import CDN_DIST
from ..models import (
from ._mixin import TooltipMixin
from .base import Widget
from .button import ButtonClick, _ClickButton
@param.depends('icon', 'active_icon', watch=True, on_init=True)
def _update_icon(self):
    if not self.icon:
        raise ValueError('The icon parameter must not be empty.')
    icon_is_svg = self.icon.startswith('<svg')
    if icon_is_svg and (not self.active_icon):
        raise ValueError('The active_icon parameter must not be empty if icon is an SVG.')