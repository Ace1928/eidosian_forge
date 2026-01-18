from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class StringEditor(CellEditor):
    """ Basic string cell editor with auto-completion.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    completions = List(String, help='\n    An optional list of completion strings.\n    ')