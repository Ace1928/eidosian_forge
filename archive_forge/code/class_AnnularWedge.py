from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class AnnularWedge(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render annular wedges.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/AnnularWedge.py'
    _args = ('x', 'y', 'inner_radius', 'outer_radius', 'start_angle', 'end_angle', 'direction')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the center of the annular wedges.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the center of the annular wedges.\n    ')
    inner_radius = DistanceSpec(default=field('inner_radius'), help='\n    The inner radii of the annular wedges.\n    ')
    outer_radius = DistanceSpec(default=field('outer_radius'), help='\n    The outer radii of the annular wedges.\n    ')
    start_angle = AngleSpec(default=field('start_angle'), help='\n    The angles to start the annular wedges, as measured from the horizontal.\n    ')
    end_angle = AngleSpec(default=field('end_angle'), help='\n    The angles to end the annular wedges, as measured from the horizontal.\n    ')
    direction = Enum(Direction, default=Direction.anticlock, help='\n    Which direction to stroke between the start and end angles.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the annular wedges.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the annular wedges.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the annular wedges.\n    ')