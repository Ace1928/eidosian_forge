from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property_aliases import BorderRadius, Padding, TextAnchor
from ...core.property_mixins import (
from ..common.properties import Coordinate
from .annotation import Annotation, DataAnnotation
class LabelSet(DataAnnotation):
    """ Render multiple text labels as annotations.

    ``LabelSet`` will render multiple text labels at given ``x`` and ``y``
    coordinates, which can be in either screen (pixel) space, or data (axis
    range) space. In this case (as opposed to the single ``Label`` model),
    ``x`` and ``y`` can also be the name of a column from a
    :class:`~bokeh.models.sources.ColumnDataSource`, in which case the labels
    will be "vectorized" using coordinate values from the specified columns.

    The label can also be configured with a screen space offset from ``x`` and
    ``y``, by using the ``x_offset`` and ``y_offset`` properties. These offsets
    may be vectorized by giving the name of a data source column.

    Additionally, the label can be rotated with the ``angle`` property (which
    may also be a column name.)

    There are also standard text, fill, and line properties to control the
    appearance of the text, its background, as well as the rectangular bounding
    box border.

    The data source is provided by setting the ``source`` property.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates to locate the text anchors.\n    ')
    x_units = Enum(CoordinateUnits, default='data', help='\n    The unit type for the ``xs`` attribute. Interpreted as |data units| by\n    default.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates to locate the text anchors.\n    ')
    y_units = Enum(CoordinateUnits, default='data', help='\n    The unit type for the ``ys`` attribute. Interpreted as |data units| by\n    default.\n    ')
    text = NullStringSpec(default=field('text'), help='\n    The text values to render.\n    ')
    angle = AngleSpec(default=0, help='\n    The angles to rotate the text, as measured from the horizontal.\n    ')
    x_offset = NumberSpec(default=0, help='\n    Offset values to apply to the x-coordinates.\n\n    This is useful, for instance, if it is desired to "float" text a fixed\n    distance in |screen units| from a given data position.\n    ')
    y_offset = NumberSpec(default=0, help='\n    Offset values to apply to the y-coordinates.\n\n    This is useful, for instance, if it is desired to "float" text a fixed\n    distance in |screen units| from a given data position.\n    ')
    text_props = Include(TextProps, help='\n    The {prop} values for the text.\n    ')
    background_props = Include(FillProps, prefix='background', help='\n    The {prop} values for the text bounding box.\n    ')
    background_fill_color = Override(default=None)
    border_props = Include(LineProps, prefix='border', help='\n    The {prop} values for the text bounding box.\n    ')
    border_line_color = Override(default=None)