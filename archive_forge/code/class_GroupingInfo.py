from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class GroupingInfo(Model):
    """Describes how to calculate totals and sub-totals

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    getter = String('', help='\n    References the column which generates the unique keys of this sub-total (groupby).\n    ')
    aggregators = List(Instance(RowAggregator), help='\n    Describes how to aggregate the columns which will populate this sub-total.\n    ')
    collapsed = Bool(False, help='\n    Whether the corresponding sub-total is expanded or collapsed by default.\n    ')