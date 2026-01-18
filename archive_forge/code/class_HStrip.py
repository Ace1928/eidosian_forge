from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class HStrip(LineGlyph, FillGlyph, HatchGlyph):
    """ Horizontal strips of infinite width. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HStrip.py'
    _args = ('y0', 'y1')
    y0 = NumberSpec(default=field('y0'), help='\n    The y-coordinates of the coordinates of one side of the strips.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates of the coordinates of the other side of the strips.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the strips.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the strips.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the strips.\n    ')