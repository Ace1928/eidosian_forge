from __future__ import annotations
import logging # isort:skip
from ..core.enums import JitterRandomDistribution, StepMode
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .sources import ColumnarDataSource
class Dodge(Transform):
    """ Apply either fixed dodge amount to data.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = Float(default=0, help='\n    The amount to dodge the input data.\n    ')
    range = Nullable(Instance('bokeh.models.ranges.Range'), help='\n    When applying ``Dodge`` to categorical data values, the corresponding\n    ``FactorRange`` must be supplied as the ``range`` property.\n    ')