from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.enums import RenderLevel
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
from ..coordinates import CoordinateMapping
@abstract
class DataRenderer(Renderer):
    """ An abstract base class for data renderer types (e.g. ``GlyphRenderer``, ``GraphRenderer``).

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    level = Override(default='glyph')