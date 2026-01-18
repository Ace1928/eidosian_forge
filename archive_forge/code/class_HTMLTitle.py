from __future__ import annotations
import logging # isort:skip
from ....core.enums import (
from ....core.properties import (
from ....core.property_aliases import BorderRadius, Padding
from ....core.property_mixins import (
from ..annotation import DataAnnotation
from .html_annotation import HTMLAnnotation
class HTMLTitle(HTMLTextAnnotation):
    """ Render a single title box as an annotation.

    See :ref:`ug_basic_annotations_titles` for information on plotting titles.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    text = String(default='', help='\n    The text value to render.\n    ')
    vertical_align = Enum(VerticalAlign, default='bottom', help='\n    Alignment of the text in its enclosing space, *across* the direction of the text.\n    ')
    align = Enum(TextAlign, default='left', help='\n    Alignment of the text in its enclosing space, *along* the direction of the text.\n    ')
    text_line_height = Float(default=1.0, help='\n    How much additional space should be allocated for the title. The value is provided\n    as a number, but should be treated as a percentage of font size. The default is\n    100%, which means no additional space will be used.\n    ')
    offset = Float(default=0, help='\n    Offset the text by a number of pixels (can be positive or negative). Shifts the text in\n    different directions based on the location of the title:\n\n        * above: shifts title right\n        * right: shifts title down\n        * below: shifts title right\n        * left: shifts title up\n\n    ')
    standoff = Float(default=10, help='\n    ')
    text_font = String(default='helvetica', help="\n    Name of a font to use for rendering text, e.g., ``'times'``,\n    ``'helvetica'``.\n\n    ")
    text_font_size = String(default='13px')
    text_font_style = Enum(FontStyle, default='bold', help="\n    A style to use for rendering text.\n\n    Acceptable values are:\n\n    - ``'normal'`` normal text\n    - ``'italic'`` *italic text*\n    - ``'bold'`` **bold text**\n\n    ")
    text_color = Color(default='#444444', help='\n    A color to use to fill text with.\n    ')
    text_outline_color = Nullable(Color, default=None, help='\n    A color to use to fill text with.\n    ')
    text_alpha = Alpha(help='\n    An alpha value to use to fill text with.\n    ')