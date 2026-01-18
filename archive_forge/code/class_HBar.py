from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class HBar(LRTBGlyph):
    """ Render horizontal bars, given a center coordinate, ``height`` and
    (``left``, ``right``) coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HBar.py'
    _args = ('y', 'height', 'right', 'left')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the centers of the horizontal bars.\n    ')
    height = DistanceSpec(default=1, help='\n    The heights of the vertical bars.\n    ')
    left = NumberSpec(default=0, help='\n    The x-coordinates of the left edges.\n    ')
    right = NumberSpec(default=field('right'), help='\n    The x-coordinates of the right edges.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the horizontal bars.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the horizontal bars.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the horizontal bars.\n    ')