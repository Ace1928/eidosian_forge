from __future__ import annotations
import logging # isort:skip
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
from ..callbacks import Callback
from .ui_element import UIElement
class ActionItem(MenuItem):
    """ A basic menu item with an icon, label, shortcut, sub-menu and an associated action.

    Only label is required. All other properties are optional.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    icon = Nullable(Either(Image, Enum(ToolIcon), Regex('^--'), Regex('^\\.')), help='\n    An optional icon to display left of the label.\n    ')
    label = Required(String, help='\n    A plain text string label.\n    ')
    shortcut = Nullable(String, default=None, help="\n    An optional string representing the keyboard sequence triggering the action.\n\n    .. note::\n        This is only a UI hint for the user. Menus on their own don't implement\n        any support for triggering actions based on keyboard inputs.\n    ")
    menu = Nullable(Instance(lambda: Menu), default=None, help='\n    An optional sub-menu showed when hovering over this item.\n    ')
    tooltip = Nullable(String, default=None, help='\n    An optional plain text description showed when hovering over this item.\n    ')
    disabled = Bool(default=False, help='\n    Indicates whether clicking on the item activates the associated action.\n    ')
    action = Nullable(Instance(Callback), default=None, help='\n    An optional action (callback) associated with this item.\n    ')