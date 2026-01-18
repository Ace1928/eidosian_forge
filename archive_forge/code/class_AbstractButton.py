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
class AbstractButton(Widget, ButtonLike):
    """ A base class that defines common properties for all button types.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    label = Either(Instance(DOMNode), String, default='Button', help='\n    Either HTML or plain text label for the button to display.\n    ')
    icon = Nullable(Instance(Icon), help="\n    An optional image appearing to the left of button's text. An instance of\n    :class:`~bokeh.models.Icon` (such as :class:`~bokeh.models.BuiltinIcon`,\n    :class:`~bokeh.models.SVGIcon`, or :class:`~bokeh.models.TablerIcon`).`\n    ")