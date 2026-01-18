from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class ScientificFormatter(StringFormatter):
    """ Display numeric values from continuous ranges as "basic numbers",
    using scientific notation when appropriate by default.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    precision = Int(10, help='\n    How many digits of precision to display.\n    ')
    power_limit_high = Int(5, help='\n    Limit the use of scientific notation to when::\n        log(x) >= power_limit_high\n    ')
    power_limit_low = Int(-3, help='\n    Limit the use of scientific notation to when::\n        log(x) <= power_limit_low\n    ')