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
class MathTextGlyph(Text):
    """ Base class for math text glyphs.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)