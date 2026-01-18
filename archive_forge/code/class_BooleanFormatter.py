from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class BooleanFormatter(CellFormatter):
    """ Boolean (check mark) cell formatter.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    icon = Enum('check', 'check-circle', 'check-circle-o', 'check-square', 'check-square-o', help='\n    The icon visualizing the check mark.\n    ')