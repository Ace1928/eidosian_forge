from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import (
from .buttons import ButtonLike
from .widget import Widget
@abstract
class ToggleInputGroup(AbstractGroup):
    """ Abstract base class for groups with items rendered as check/radio
    boxes.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    inline = Bool(False, help='\n    Should items be arrange vertically (``False``) or horizontally\n    in-line (``True``).\n    ')