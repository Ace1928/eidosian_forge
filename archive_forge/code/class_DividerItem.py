from __future__ import annotations
import logging # isort:skip
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
from ..callbacks import Callback
from .ui_element import UIElement
class DividerItem(MenuItem):
    """ A dividing line between two groups of menu items. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)