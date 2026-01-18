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
class InputWidget(Widget):
    """ Abstract base class for input widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    title = Either(String, Instance(HTML), default='', help="\n    Widget's label.\n    ")
    description = Nullable(Either(String, Instance(Tooltip)), default=None, help='\n    Either a plain text or a tooltip with a rich HTML description of the function of this widget.\n    ')