from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable
from ...core.enums import ButtonType
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from ...events import ButtonClick, MenuItemClick
from ..callbacks import Callback
from ..dom import DOMNode
from ..ui.icons import BuiltinIcon, Icon
from ..ui.tooltips import Tooltip
from .widget import Widget
@abstract
class ButtonLike(HasProps):
    """ Shared properties for button-like widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    button_type = Enum(ButtonType, help="\n    A style for the button, signifying it's role. Possible values are one of the\n    following:\n\n    .. bokeh-plot::\n        :source-position: none\n\n        from bokeh.core.enums import ButtonType\n        from bokeh.io import show\n        from bokeh.layouts import column\n        from bokeh.models import Button\n\n        show(column(\n            [Button(label=button_type, button_type=button_type) for button_type in ButtonType]\n            ))\n    ")