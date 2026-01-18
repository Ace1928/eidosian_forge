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
class TextLikeInput(InputWidget):
    """ Base class for text-like input widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = String(default='', help='\n    Initial or entered text value.\n\n    Change events are triggered whenever <enter> is pressed.\n    ')
    value_input = String(default='', help='\n    Initial or current value.\n\n    Change events are triggered whenever any update happens, i.e. on every\n    keypress.\n    ')
    placeholder = String(default='', help='\n    Placeholder for empty input field.\n    ')
    max_length = Nullable(Int, help='\n    Max count of characters in field\n    ')