from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Instance, List
from ..model import Model
from .graphics import Decoration
@abstract
class ConnectedXYGlyph(XYGlyph):
    """ Base class of glyphs with `x` and `y` coordinate attributes and
    a connected topology.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)