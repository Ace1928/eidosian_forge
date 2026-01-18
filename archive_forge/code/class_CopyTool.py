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
class CopyTool(ActionTool):
    """ *toolbar icon*: |copy_icon|

    The copy tool is an action tool, that allows copying the rendererd contents of
    a plot or a collection of plots to system's clipboard. This tools is browser
    dependent and may not function in certain browsers, or require additional
    permissions to be granted to the web page.

    .. |copy_icon| image:: /_images/icons/Copy.png
        :height: 24px

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)