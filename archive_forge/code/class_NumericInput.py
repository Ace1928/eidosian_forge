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
class NumericInput(InputWidget):
    """ Numeric input widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = Either(Null, Float, Int, help='\n    Initial or entered value.\n\n    Change events are triggered whenever <enter> is pressed.\n    ')
    low = Either(Null, Float, Int, help='\n    Optional lowest allowable value.\n    ')
    high = Either(Null, Float, Int, help='\n    Optional highest allowable value.\n    ')
    placeholder = String(default='', help='\n    Placeholder for empty input field.\n    ')
    mode = Enum('int', 'float', help='\n    Define the type of number which can be enter in the input\n\n    example\n    mode int: 1, -1, 156\n    mode float: 1, -1.2, 1.1e-25\n    ')
    format = Either(Null, String, Instance(TickFormatter), help='\n    ')