from __future__ import annotations
import logging # isort:skip
from ..core.enums import JitterRandomDistribution, StepMode
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .sources import ColumnarDataSource
class Jitter(Transform):
    """ Apply either a uniform or normally sampled random jitter to data.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    mean = Float(default=0, help='\n    The central value for the random sample\n    ')
    width = Float(default=1, help='\n    The width (absolute for uniform distribution and sigma for the normal\n    distribution) of the random sample.\n    ')
    distribution = Enum(JitterRandomDistribution, default='uniform', help='\n    The random distribution upon which to pull the random scatter\n    ')
    range = Nullable(Instance('bokeh.models.ranges.Range'), help='\n    When applying Jitter to Categorical data values, the corresponding\n    ``FactorRange`` must be supplied as the ``range`` property.\n    ')