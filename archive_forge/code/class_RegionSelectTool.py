from __future__ import annotations
import logging # isort:skip
import difflib
import typing as tp
from math import nan
from typing import Literal
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.validation import error
from ..core.validation.errors import NO_RANGE_TOOL_RANGES
from ..model import Model
from ..util.strings import nice_join
from .annotations import BoxAnnotation, PolyAnnotation, Span
from .callbacks import Callback
from .dom import Template
from .glyphs import (
from .nodes import Node
from .ranges import Range
from .renderers import DataRenderer, GlyphRenderer
from .ui import UIElement
@abstract
class RegionSelectTool(SelectTool):
    """ Base class for region selection tools (e.g. box, polygon, lasso).

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    continuous = Bool(False, help='\n    Whether a selection computation should happen continuously during selection\n    gestures, or only once when the selection region is completed.\n    ')
    select_every_mousemove = DeprecatedAlias('continuous', since=(3, 1, 0))
    persistent = Bool(default=False, help='\n    Whether the selection overlay should persist after selection gesture\n    is completed. This can be paired with setting ``editable = True`` on\n    the annotation, to allow to modify the selection.\n    ')
    greedy = Bool(default=False, help='\n    Defines whether a hit against a glyph requires full enclosure within\n    the selection region (non-greedy) or only an intersection (greedy)\n    (i.e. at least one point within the region).\n    ')