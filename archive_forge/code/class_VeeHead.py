from __future__ import annotations
import logging # isort:skip
from ...core.enums import CoordinateUnits
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property_mixins import FillProps, LineProps
from ..graphics import Marking
from .annotation import DataAnnotation
class VeeHead(ArrowHead):
    """ Render a vee-style arrow head.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    line_props = Include(LineProps, help='\n    The {prop} values for the arrow head outline.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the arrow head interior.\n    ')
    fill_color = Override(default='black')