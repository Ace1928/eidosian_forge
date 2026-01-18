from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class ScalarFillProps(HasProps):
    """ Properties relevant to rendering fill regions.

    Mirrors the BokehJS ``properties.Fill`` class.

    """
    fill_color = Nullable(Color, default='gray', help=_color_help % 'fill paths')
    fill_alpha = Alpha(help=_alpha_help % 'fill paths')