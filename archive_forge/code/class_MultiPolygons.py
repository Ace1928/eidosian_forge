from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class MultiPolygons(LineGlyph, FillGlyph, HatchGlyph):
    """ Render several MultiPolygon.

    Modeled on geoJSON - the data for the ``MultiPolygons`` glyph is
    different in that the vector of values is not a vector of scalars.
    Rather, it is a "list of lists of lists of lists".

    During box selection only multi-polygons entirely contained in the
    selection box will be included.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/MultiPolygons.py'
    _args = ('xs', 'ys')
    xs = NumberSpec(default=field('xs'), help='\n    The x-coordinates for all the patches, given as a nested list.\n\n    .. note::\n        Each item in ``MultiPolygons`` represents one MultiPolygon and each\n        MultiPolygon is comprised of ``n`` Polygons. Each Polygon is made of\n        one exterior ring optionally followed by ``m`` interior rings (holes).\n    ')
    ys = NumberSpec(default=field('ys'), help='\n    The y-coordinates for all the patches, given as a "list of lists".\n\n    .. note::\n        Each item in ``MultiPolygons`` represents one MultiPolygon and each\n        MultiPolygon is comprised of ``n`` Polygons. Each Polygon is made of\n        one exterior ring optionally followed by ``m`` interior rings (holes).\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the multipolygons.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the multipolygons.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the multipolygons.\n    ')