from __future__ import annotations
import logging # isort:skip
from ..core.enums import JitterRandomDistribution, StepMode
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .sources import ColumnarDataSource
class StepInterpolator(Interpolator):
    """ Compute a step-wise interpolation between the points provided through
    the ``x``, ``y``, and ``data`` parameters.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    mode = Enum(StepMode, default='after', help='\n    Adjust the behavior of the returned value in relation to the control points.  The parameter can assume one of three values:\n\n    * ``after`` (default): Assume the y-value associated with the nearest x-value which is less than or equal to the point to transform.\n    * ``before``: Assume the y-value associated with the nearest x-value which is greater than the point to transform.\n    * ``center``: Assume the y-value associated with the nearest x-value to the point to transform.\n    ')