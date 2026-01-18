from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class DataCube(DataTable):
    """Specialized DataTable with collapsing groups, totals, and sub-totals.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    grouping = List(Instance(GroupingInfo), help='\n    Describe what aggregation operations used to define sub-totals and totals\n    ')
    target = Instance(DataSource, help='\n    Two column datasource (row_indices & labels) describing which rows of the\n    data cubes are expanded or collapsed\n    ')