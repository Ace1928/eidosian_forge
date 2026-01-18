from __future__ import annotations
import logging # isort:skip
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.bases import Init
from ...core.property.singletons import Intrinsic
from .ui_element import UIElement
class SVGIcon(Icon):
    """ SVG icons with inline definitions. """

    def __init__(self, svg: Init[str]=Intrinsic, **kwargs) -> None:
        super().__init__(svg=svg, **kwargs)
    svg = Required(String, help='\n    The SVG definition of an icon.\n    ')