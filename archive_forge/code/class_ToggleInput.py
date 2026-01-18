from __future__ import annotations
import logging # isort:skip
from math import inf
from typing import Any as any
from ...core.has_props import abstract
from ...core.properties import (
from ...util.deprecation import deprecated
from ..dom import HTML
from ..formatters import TickFormatter
from ..ui import Tooltip
from .widget import Widget
@abstract
class ToggleInput(Widget):
    """ Base class for toggleable (boolean) input widgets. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    active = Bool(default=False, help='\n    The state of the widget.\n    ')