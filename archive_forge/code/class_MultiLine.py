from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class MultiLine(LineGlyph):
    """ Render several lines.

    The data for the ``MultiLine`` glyph is different in that the vector of
    values is not a vector of scalars. Rather, it is a "list of lists".

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/MultiLine.py'
    _args = ('xs', 'ys')
    xs = NumberSpec(default=field('xs'), help='\n    The x-coordinates for all the lines, given as a "list of lists".\n    ')
    ys = NumberSpec(default=field('ys'), help='\n    The y-coordinates for all the lines, given as a "list of lists".\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the lines.\n    ')