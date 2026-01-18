from __future__ import annotations
from typing import (
import param
from ..io.resources import CDN_DIST
from ..models import (
from ._mixin import TooltipMixin
from .base import Widget
from .button import ButtonClick, _ClickButton
class ToggleIcon(_ClickableIcon, TooltipMixin):
    """
    The `ToggleIcon` widget allows toggling a single condition between True/False states. This
    widget is interchangeable with the `Checkbox` and `Toggle` widget.

    This widget incorporates a `value` attribute, which alternates between `False` and `True`.

    Reference: https://panel.holoviz.org/reference/widgets/ToggleIcon.html

    :Example:

    >>> pn.widgets.ToggleIcon(
    ...     icon="thumb-up", active_icon="thumb-down", size="4em", description="Like"
    ... )
    """
    _widget_type = _PnToggleIcon