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
class Toggle(AbstractButton):
    """ A two-state toggle button.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    label = Override(default='Toggle')
    active = Bool(False, help='\n    The state of the toggle button.\n    ')

    def on_click(self, handler: Callable[[bool], None]) -> None:
        """ Set up a handler for button state changes (clicks).

        Args:
            handler (func) : handler function to call when button is toggled.

        Returns:
            None
        """
        self.on_change('active', lambda attr, old, new: handler(new))

    def js_on_click(self, handler: Callback) -> None:
        """ Set up a JavaScript handler for button state changes (clicks). """
        self.js_on_change('active', handler)