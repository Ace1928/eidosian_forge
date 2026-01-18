from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Instance, List
from ..model import Model
from .graphics import Decoration
@abstract
class LineGlyph(Glyph):
    """ Glyphs with line properties

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)