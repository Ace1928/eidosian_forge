from __future__ import annotations
import logging # isort:skip
from math import inf
from ..core.enums import Direction
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class PolarTransform(CoordinateTransform):
    """ Transform from polar to cartesian coordinates. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    radius = NumberSpec(default=field('radius'), help='\n    The radial coordinate (i.e. the distance from the origin).\n\n    Negative radius is allowed, which is equivalent to using positive radius\n    and changing ``direction`` to the opposite value.\n    ')
    angle = AngleSpec(default=field('angle'), help='\n    The angular coordinate (i.e. the angle from the reference axis).\n    ')
    direction = Enum(Direction, default=Direction.anticlock, help='\n    Whether ``angle`` measures clockwise or anti-clockwise from the reference axis.\n    ')