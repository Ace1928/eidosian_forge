from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class HexTile(LineGlyph, FillGlyph, HatchGlyph):
    """ Render horizontal tiles on a regular hexagonal grid.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HexTile.py'
    _args = ('q', 'r')
    size = Float(1.0, help='\n    The radius (in |data units|) of the hex tiling.\n\n    The radius is always measured along the cartesian y-axis for "pointy_top"\n    orientation, and along the cartesian x-axis for "flat_top" orientation. If\n    the aspect ratio of the underlying cartesian system is not 1-1, then the\n    tiles may be "squished" in one direction. To ensure that the tiles are\n    always regular hexagons, consider setting the ``match_aspect`` property of\n    the plot to True.\n    ')
    aspect_scale = Float(default=1.0, help="\n    Match a plot's aspect ratio scaling.\n\n    Use this parameter to match the aspect ratio scaling of a plot when using\n    :class:`~bokeh.models.Plot.aspect_scale` with a value other than ``1.0``.\n\n    ")
    r = NumberSpec(default=field('r'), help='\n    The "row" axial coordinates of the tile centers.\n    ')
    q = NumberSpec(default=field('q'), help='\n    The "column" axial coordinates of the tile centers.\n    ')
    scale = NumberSpec(1.0, help='\n    A scale factor for individual tiles.\n    ')
    orientation = String(default='pointytop', help='\n    The orientation of the hex tiles.\n\n    Use ``"pointytop"`` to orient the tile so that a pointed corner is at the top. Use\n    ``"flattop"`` to orient the tile so that a flat side is at the top.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the hex tiles.\n    ')
    line_color = Override(default=None)
    fill_props = Include(FillProps, help='\n    The {prop} values for the hex tiles.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the hex tiles.\n    ')