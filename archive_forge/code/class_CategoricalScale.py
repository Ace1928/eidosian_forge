from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Instance, Required
from .transforms import Transform
class CategoricalScale(Scale):
    """ Represent a scale transformation between a categorical source range and
    continuous target range.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)