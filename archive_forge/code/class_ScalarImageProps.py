from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class ScalarImageProps(HasProps):
    """ Properties relevant to rendering images.

    Mirrors the BokehJS ``properties.Image`` class.

    """
    global_alpha = Alpha(help=_alpha_help % 'images')