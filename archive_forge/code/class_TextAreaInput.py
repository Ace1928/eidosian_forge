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
class TextAreaInput(TextLikeInput):
    """ Multi-line input widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    cols = Int(default=20, help='\n    Specifies the width of the text area (in average character width). Default: 20\n    ')
    rows = Int(default=2, help='\n    Specifies the height of the text area (in lines). Default: 2\n    ')
    max_length = Override(default=500)