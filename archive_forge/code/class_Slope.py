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
class Slope(Annotation):
    """ Render a sloped line as an annotation.

    See :ref:`ug_basic_annotations_slope` for information on plotting slopes.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    gradient = Nullable(Float, help='\n    The gradient of the line, in |data units|\n    ')
    y_intercept = Nullable(Float, help='\n    The y intercept of the line, in |data units|\n    ')
    line_props = Include(ScalarLineProps, help='\n    The {prop} values for the line.\n    ')
    above_fill_props = Include(ScalarFillProps, prefix='above', help='\n    The {prop} values for the area above the line.\n    ')
    above_hatch_props = Include(ScalarHatchProps, prefix='above', help='\n    The {prop} values for the area above the line.\n    ')
    below_fill_props = Include(ScalarFillProps, prefix='below', help='\n    The {prop} values for the area below the line.\n    ')
    below_hatch_props = Include(ScalarHatchProps, prefix='below', help='\n    The {prop} values for the area below the line.\n    ')
    above_fill_color = Override(default=None)
    above_fill_alpha = Override(default=0.4)
    below_fill_color = Override(default=None)
    below_fill_alpha = Override(default=0.4)