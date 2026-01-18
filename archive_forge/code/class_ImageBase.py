from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
@abstract
class ImageBase(XYGlyph):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates to locate the image anchors.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates to locate the image anchors.\n    ')
    dw = DistanceSpec(default=field('dw'), help='\n    The widths of the plot regions that the images will occupy.\n\n    .. note::\n        This is not the number of pixels that an image is wide.\n        That number is fixed by the image itself.\n    ')
    dh = DistanceSpec(default=field('dh'), help='\n    The height of the plot region that the image will occupy.\n\n    .. note::\n        This is not the number of pixels that an image is tall.\n        That number is fixed by the image itself.\n    ')
    image_props = Include(ImageProps, help='\n    The {prop} values for the images.\n    ')
    dilate = Bool(False, help='\n    Whether to always round fractional pixel locations in such a way\n    as to make the images bigger.\n\n    This setting may be useful if pixel rounding errors are causing\n    images to have a gap between them, when they should appear flush.\n    ')
    origin = Enum(ImageOrigin, default='bottom_left', help='\n    Defines the coordinate space of an image.\n    ')
    anchor = Anchor(default='bottom_left', help='\n    Position of the image should be anchored at the `x`, `y` coordinates.\n    ')