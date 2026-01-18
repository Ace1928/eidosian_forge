from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from ..util.deprecation import deprecated
from ..util.serialization import convert_datetime_array
from ..util.warnings import BokehUserWarning, warn
from .callbacks import CustomJS
from .filters import AllIndices, Filter, IntersectionFilter
from .selections import Selection, SelectionPolicy, UnionRenderers
@abstract
class ColumnarDataSource(DataSource):
    """ A base class for data source types, which can be mapped onto
    a columnar format.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    default_values = Dict(String, Any, default={}, help='\n    Defines the default value for each column.\n\n    This is used when inserting rows into a data source, e.g. by edit tools,\n    when a value for a given column is not explicitly provided. If a default\n    value is missing, a tool will defer to its own configuration or will try\n    to let the data source to infer a sensible default value.\n    ')
    selection_policy = Instance(SelectionPolicy, default=InstanceDefault(UnionRenderers), help='\n    An instance of a ``SelectionPolicy`` that determines how selections are set.\n    ')