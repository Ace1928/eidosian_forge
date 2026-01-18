from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.enums import RenderLevel
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
from ..coordinates import CoordinateMapping
@abstract
class GuideRenderer(Renderer):
    """ A base class for all guide renderer types. ``GuideRenderer`` is
    not generally useful to instantiate on its own.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    level = Override(default='guide')