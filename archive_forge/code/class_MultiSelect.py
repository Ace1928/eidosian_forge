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
class MultiSelect(InputWidget):
    """ Multi-select widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    options = List(Either(String, Tuple(String, String)), help='\n    Available selection options. Options may be provided either as a list of\n    possible string values, or as a list of tuples, each of the form\n    ``(value, label)``. In the latter case, the visible widget text for each\n    value will be corresponding given label.\n    ')
    value = List(String, help='\n    Initial or selected values.\n    ')
    size = Int(default=4, help="\n    The number of visible options in the dropdown list. (This uses the\n    ``select`` HTML element's ``size`` attribute. Some browsers might not\n    show less than 3 options.)\n    ")