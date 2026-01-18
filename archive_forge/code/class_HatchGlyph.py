from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Instance, List
from ..model import Model
from .graphics import Decoration
@abstract
class HatchGlyph(Glyph):
    """ Glyphs with Hatch properties

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)