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
class MultiChoice(InputWidget):
    """ MultiChoice widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    options = List(Either(String, Tuple(String, String)), help='\n    Available selection options. Options may be provided either as a list of\n    possible string values, or as a list of tuples, each of the form\n    ``(value, label)``. In the latter case, the visible widget text for each\n    value will be corresponding given label.\n    ')
    value = List(String, help='\n    Initial or selected values.\n    ')
    delete_button = Bool(default=True, help='\n    Whether to add a button to remove a selected option.\n    ')
    max_items = Nullable(Int, help='\n    The maximum number of items that can be selected.\n    ')
    option_limit = Nullable(Int, help='\n    The number of choices that will be rendered in the dropdown.\n    ')
    search_option_limit = Nullable(Int, help='\n    The number of choices that will be rendered in the dropdown\n    when search string is entered.\n    ')
    placeholder = Nullable(String, help='\n    A string that is displayed if not item is added.\n    ')
    solid = Bool(default=True, help='\n    Specify whether the choices should be solidly filled.')