from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.enums import RenderLevel
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
from ..coordinates import CoordinateMapping
class RendererGroup(Model):
    """A collection of renderers.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    visible = Bool(default=True, help='\n    Makes all grouped renderers visible or not.\n    ')