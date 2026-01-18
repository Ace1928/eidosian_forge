from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class TableColumn(Model):
    """ Table column widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    field = Required(String, help='\n    The name of the field mapping to a column in the data source.\n    ')
    title = Nullable(String, help="\n    The title of this column. If not set, column's data field is\n    used instead.\n    ")
    width = Int(300, help="\n    The width or maximum width (depending on data table's configuration)\n    in pixels of this column.\n    ")
    formatter = Instance(CellFormatter, InstanceDefault(StringFormatter), help='\n    The cell formatter for this column. By default, a simple string\n    formatter is used.\n    ')
    editor = Instance(CellEditor, InstanceDefault(StringEditor), help='\n    The cell editor for this column. By default, a simple string editor\n    is used.\n    ')
    sortable = Bool(True, help='\n    Whether this column is sortable or not. Note that data table has\n    to have sorting enabled to allow sorting in general.\n    ')
    default_sort = Enum('ascending', 'descending', help='\n    The default sorting order. By default ``ascending`` order is used.\n    ')
    visible = Bool(True, help='\n    Whether this column shold be displayed or not.\n    ')