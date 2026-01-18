from __future__ import annotations
import logging # isort:skip
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Literal
from bokeh.core.property.vectorization import Field
from ...core.properties import (
from ...core.validation import error
from ...core.validation.errors import BAD_COLUMN_NAME, CDSVIEW_FILTERS_WITH_CONNECTED
from ..filters import AllIndices
from ..glyphs import ConnectedXYGlyph, Glyph
from ..graphics import Decoration, Marking
from ..sources import (
from .renderer import DataRenderer
def add_decoration(self, marking: Marking, node: Literal['start', 'middle', 'end']) -> Decoration:
    glyphs = [self.glyph, self.selection_glyph, self.nonselection_glyph, self.hover_glyph, self.muted_glyph]
    decoration = Decoration(marking=marking, node=node)
    for glyph in glyphs:
        if isinstance(glyph, Glyph):
            glyph.decorations.append(decoration)
    return decoration