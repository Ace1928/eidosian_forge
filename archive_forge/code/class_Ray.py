from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class Ray(XYGlyph, LineGlyph):
    """ Render rays.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Ray.py'
    _args = ('x', 'y', 'length', 'angle')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates to start the rays.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates to start the rays.\n    ')
    angle = AngleSpec(default=0, help='\n    The angles in radians to extend the rays, as measured from the horizontal.\n    ')
    length = DistanceSpec(default=0, help='\n    The length to extend the ray. Note that this ``length`` defaults\n    to |data units| (measured in the x-direction).\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the rays.\n    ')