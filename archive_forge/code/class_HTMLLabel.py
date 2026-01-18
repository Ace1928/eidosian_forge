from __future__ import annotations
import logging # isort:skip
from ....core.enums import (
from ....core.properties import (
from ....core.property_aliases import BorderRadius, Padding
from ....core.property_mixins import (
from ..annotation import DataAnnotation
from .html_annotation import HTMLAnnotation
class HTMLLabel(HTMLTextAnnotation):
    """ Render a single HTML label as an annotation.

    ``Label`` will render a single text label at given ``x`` and ``y``
    coordinates, which can be in either screen (pixel) space, or data (axis
    range) space.

    The label can also be configured with a screen space offset from ``x`` and
    ``y``, by using the ``x_offset`` and ``y_offset`` properties.

    Additionally, the label can be rotated with the ``angle`` property.

    There are also standard text, fill, and line properties to control the
    appearance of the text, its background, as well as the rectangular bounding
    box border.

    See :ref:`ug_basic_annotations_labels` for information on plotting labels.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    x = Required(CoordinateLike, help='\n    The x-coordinate in screen coordinates to locate the text anchors.\n    ')
    x_units = Enum(CoordinateUnits, default='data', help='\n    The unit type for the x attribute. Interpreted as |data units| by\n    default.\n    ')
    y = Required(CoordinateLike, help='\n    The y-coordinate in screen coordinates to locate the text anchors.\n    ')
    y_units = Enum(CoordinateUnits, default='data', help='\n    The unit type for the y attribute. Interpreted as |data units| by\n    default.\n    ')
    text = String(default='', help='\n    The text value to render.\n    ')
    angle = Angle(default=0, help='\n    The angle to rotate the text, as measured from the horizontal.\n    ')
    angle_units = Enum(AngleUnits, default='rad', help='\n    Acceptable values for units are ``"rad"`` and ``"deg"``\n    ')
    x_offset = Float(default=0, help='\n    Offset value to apply to the x-coordinate.\n\n    This is useful, for instance, if it is desired to "float" text a fixed\n    distance in |screen units| from a given data position.\n    ')
    y_offset = Float(default=0, help='\n    Offset value to apply to the y-coordinate.\n\n    This is useful, for instance, if it is desired to "float" text a fixed\n    distance in |screen units| from a given data position.\n    ')
    text_props = Include(ScalarTextProps, help='\n    The {prop} values for the text.\n    ')