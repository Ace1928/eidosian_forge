from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class HArea(LineGlyph, FillGlyph, HatchGlyph):
    """ Render a horizontally directed area between two equal length sequences
    of x-coordinates with the same y-coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HArea.py'
    _args = ('x1', 'x2', 'y')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates for the points of one side of the area.\n    ')
    x2 = NumberSpec(default=field('x2'), help='\n    The x-coordinates for the points of the other side of the area.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates for the points of the area.\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the horizontal directed area.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the horizontal directed area.\n    ')