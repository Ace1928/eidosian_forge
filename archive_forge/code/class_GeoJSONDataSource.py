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
class GeoJSONDataSource(ColumnarDataSource):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    geojson = Required(JSON, help='\n    GeoJSON that contains features for plotting. Currently\n    ``GeoJSONDataSource`` can only process a ``FeatureCollection`` or\n    ``GeometryCollection``.\n    ')