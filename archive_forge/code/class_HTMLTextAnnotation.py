from __future__ import annotations
import logging # isort:skip
from ....core.enums import (
from ....core.properties import (
from ....core.property_aliases import BorderRadius, Padding
from ....core.property_mixins import (
from ..annotation import DataAnnotation
from .html_annotation import HTMLAnnotation
class HTMLTextAnnotation(HTMLAnnotation):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    padding = Padding(default=0, help='\n    Extra space between the text of a label and its bounding box (border).\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    border_radius = BorderRadius(default=0, help="\n    Allows label's box to have rounded corners. For the best results, it\n    should be used in combination with ``padding``.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ")
    background_props = Include(ScalarFillProps, prefix='background', help='\n    The {prop} values for the text bounding box.\n    ')
    background_hatch_props = Include(ScalarHatchProps, prefix='background', help='\n    The {prop} values for the text bounding box.\n    ')
    border_props = Include(ScalarLineProps, prefix='border', help='\n    The {prop} values for the text bounding box.\n    ')
    background_fill_color = Override(default=None)
    background_hatch_color = Override(default=None)
    border_line_color = Override(default=None)