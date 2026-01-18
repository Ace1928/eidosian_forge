from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class GroupFilter(Filter):
    """ A ``GroupFilter`` represents the rows of a ``ColumnDataSource`` where the values of the categorical
    column column_name match the group variable.
    """
    column_name = Required(String, help='\n    The name of the column to perform the group filtering operation on.\n    ')
    group = Required(String, help='\n    The value of the column indicating the rows of data to keep.\n    ')

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 2 and 'column_name' not in kwargs and ('group' not in kwargs):
            kwargs['column_name'] = args[0]
            kwargs['group'] = args[1]
        super().__init__(**kwargs)