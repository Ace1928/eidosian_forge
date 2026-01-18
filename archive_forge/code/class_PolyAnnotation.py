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
class PolyAnnotation(Annotation):
    """ Render a shaded polygonal region as an annotation.

    See :ref:`ug_basic_annotations_polygon_annotations` for information on
    plotting polygon annotations.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    xs = Seq(CoordinateLike, default=[], help='\n    The x-coordinates of the region to draw.\n    ')
    xs_units = Enum(CoordinateUnits, default='data', help='\n    The unit type for the ``xs`` attribute. Interpreted as |data units| by\n    default.\n    ')
    ys = Seq(CoordinateLike, default=[], help='\n    The y-coordinates of the region to draw.\n    ')
    ys_units = Enum(CoordinateUnits, default='data', help='\n    The unit type for the ``ys`` attribute. Interpreted as |data units| by\n    default.\n    ')
    editable = Bool(default=False, help='\n    Allows to interactively modify the geometry of this polygon.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    line_props = Include(ScalarLineProps, help='\n    The {prop} values for the polygon.\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the polygon.\n    ')
    hatch_props = Include(ScalarHatchProps, help='\n    The {prop} values for the polygon.\n    ')
    hover_line_props = Include(ScalarLineProps, prefix='hover', help='\n    The {prop} values for the polygon when hovering over.\n    ')
    hover_fill_props = Include(ScalarFillProps, prefix='hover', help='\n    The {prop} values for the polygon when hovering over.\n    ')
    hover_hatch_props = Include(ScalarHatchProps, prefix='hover', help='\n    The {prop} values for the polygon when hovering over.\n    ')
    line_color = Override(default='#cccccc')
    line_alpha = Override(default=0.3)
    fill_color = Override(default='#fff9ba')
    fill_alpha = Override(default=0.4)
    hover_line_color = Override(default=None)
    hover_line_alpha = Override(default=0.3)
    hover_fill_color = Override(default=None)
    hover_fill_alpha = Override(default=0.4)