from __future__ import annotations
import logging # isort:skip
from typing import Any, ClassVar, Literal
from ..core.properties import (
from ..core.property.aliases import CoordinateLike
from ..model import Model
class XY(Coordinate):
    """ A point in a Cartesian coordinate system.

    .. note::
        This model is experimental and may change at any point.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
    x = Required(CoordinateLike, help='\n    The x component.\n    ')
    y = Required(CoordinateLike, help='\n    The y component.\n    ')