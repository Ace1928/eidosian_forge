from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class Patches(LineGlyph, FillGlyph, HatchGlyph):
    """ Render several patches.

    The data for the ``Patches`` glyph is different in that the vector of
    values is not a vector of scalars. Rather, it is a "list of lists".

    During box selection only patches entirely contained in the
    selection box will be included.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Patches.py'
    _args = ('xs', 'ys')
    xs = NumberSpec(default=field('xs'), help='\n    The x-coordinates for all the patches, given as a "list of lists".\n\n    .. note::\n        Individual patches may comprise multiple polygons. In this case\n        the x-coordinates for each polygon should be separated by NaN\n        values in the sublists.\n    ')
    ys = NumberSpec(default=field('ys'), help='\n    The y-coordinates for all the patches, given as a "list of lists".\n\n    .. note::\n        Individual patches may comprise multiple polygons. In this case\n        the y-coordinates for each polygon should be separated by NaN\n        values in the sublists.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the patches.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the patches.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the patches.\n    ')