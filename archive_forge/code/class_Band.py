from __future__ import annotations
import logging # isort:skip
from math import inf
from ...core.enums import (
from ...core.properties import (
from ...core.property_aliases import BorderRadius
from ...core.property_mixins import (
from ..common.properties import Coordinate
from ..nodes import BoxNodes, Node
from .annotation import Annotation, DataAnnotation
from .arrows import ArrowHead, TeeHead
class Band(DataAnnotation):
    """ Render a filled area band along a dimension.

    See :ref:`ug_basic_annotations_bands` for information on plotting bands.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    lower = UnitsSpec(default=field('lower'), units_enum=CoordinateUnits, units_default='data', help='\n    The coordinates of the lower portion of the filled area band.\n    ')
    upper = UnitsSpec(default=field('upper'), units_enum=CoordinateUnits, units_default='data', help='\n    The coordinates of the upper portion of the filled area band.\n    ')
    base = UnitsSpec(default=field('base'), units_enum=CoordinateUnits, units_default='data', help='\n    The orthogonal coordinates of the upper and lower values.\n    ')
    dimension = Enum(Dimension, default='height', help='\n    The direction of the band can be specified by setting this property\n    to "height" (``y`` direction) or "width" (``x`` direction).\n    ')
    line_props = Include(ScalarLineProps, help='\n    The {prop} values for the band.\n    ')
    line_alpha = Override(default=0.3)
    line_color = Override(default='#cccccc')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the band.\n    ')
    fill_alpha = Override(default=0.4)
    fill_color = Override(default='#fff9ba')